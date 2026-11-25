"""Part 4 (training), the corpus: public documents of FAU, cleaned, for fine-tuning GPT-2.

Run from the repository root:

    .venv/bin/python lectures/05_self_attention_and_transformers/figures/part4_fau_corpus.py [preview_dir] [--crawl] [--clean]

Two stages, each written to data/fau/ next to the slides (git-ignored, never committed):

1. crawl (only with --crawl; without it the script never touches the network): a polite crawler that
   reads the sitemaps of a few public FAU web sites and follows links to other FAU hosts,
   fetching HTML pages and the PDF files they link to (module handbooks, study guides,
   regulations).  It reads robots.txt of every host and obeys it, waits 0.6 s between two
   requests to the same host (at most 1.7 requests per second per host), sends a generic
   User-Agent and nothing else, never logs in, and stops after CRAWL_MINUTES (environment
   variable, default 40).  An interrupted crawl resumes: what is in fetched.jsonl is not fetched
   again.  Most FAU sites (www.fau.eu, www.fau.de, the faculties, most departments and the
   study-programme sites) answer automated requests with a proof-of-work page ("Anubis"): such a
   host is recorded as blocked and left alone, the crawler does not try to get past it.  Raw
   responses go to data/fau/raw/, one line per fetch to data/fau/raw/fetched.jsonl.
   The corpus of the slides: a crawl on 2026-11-25, 21:14-21:33 CET (19 minutes, stopped by
   hand to leave the CPU to the training), 1 657 documents.

2. clean (always, unless the corpus is already there; --clean forces it): text from HTML
   (the main content only: navigation, header, footer, cookie banners, side bars and lists
   that are only links removed) and from PDFs (pdftotext in reading order; running headers and
   footers, page numbers, tables of contents with dot leaders and hyphenation at line ends
   removed); typographic quotes and dashes made plain; e-mail addresses and telephone numbers
   removed; research papers, theses, publication lists and staff pages dropped; the language of
   every paragraph detected from function words; a paragraph that occurs in more than one
   document kept only once (the same boilerplate is on thousands of pages); documents that are
   left with less than 300 characters dropped.  Documents are split 95/5 into training and
   validation by a hash of their URL, so that the validation text comes from documents never
   trained on.  Output: data/fau/corpus/{en,de}_{train,val}.txt (documents separated by a line
   <|endoftext|>), manifest.jsonl (URL, host, type, language, split, characters, words, tokens,
   date fetched) and stats.json.

Printed: the hosts crawled and blocked, and the size of the corpus in documents, characters,
words and GPT-2 tokens, per source type and language.  The cleaning takes about five minutes on
a loaded machine (most of it parsing HTML).
"""
import collections
import datetime
import gzip
import hashlib
import heapq
import json
import multiprocessing
import pathlib
import random
import re
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.robotparser

import common as C

HERE = pathlib.Path(__file__).resolve().parent
FAU = C.DATA / "fau"
RAW = FAU / "raw"
CORPUS = FAU / "corpus"

# ---------------------------------------------------------------------------------------
# 1. The crawl
# ---------------------------------------------------------------------------------------
USER_AGENT = "python-requests/2.34"         # generic: nothing about this project or its people
DELAY = 0.6                                 # seconds between two requests to the same host
CRAWL_MINUTES = int(__import__('os').environ.get('CRAWL_MINUTES', 40))
MAX_HOSTS = 40                              # hosts that answer (those that ask for a proof of work do not count)
MAX_PAGES_PER_HOST, MAX_PDFS_PER_HOST = 600, 80
MAX_PAGES, MAX_PDFS = 6000, 400
MAX_PDF_BYTES, MAX_HTML_BYTES = 20_000_000, 3_000_000

# Where the crawl starts: the public sites of FAU that answered plain requests in November 2026
# (the central sites www.fau.eu / www.fau.de and most faculty sites ask for a proof of work and
# are not crawled).  Their sitemaps list the pages; links lead to further FAU hosts.
SEEDS = [
    "https://en.www.math.fau.de/",           # Department of Mathematics, English site
    "https://www.math.fau.de/",              # the same in German
    "https://lme.tf.fau.de/",                # Pattern Recognition Lab (English)
    "https://ub.fau.de/en/",                 # University Library (English and German)
    "https://sys.cs.fau.de/",                # Chair of Computer Science 4 (systems software)
    "https://cs.fau.de/",                    # Department of Computer Science (German)
    "https://www.zuv.fau.de/",               # central administration (German)
]
# Hosts never visited: logins, single sign-on, course management, mail, video, repositories,
# and the servers of personal home pages (students' and staff's own pages are not FAU documents).
EXCLUDED_HOSTS = re.compile(r"(^|[.-])(studon|campo|univis|idm|sso|login|mail|webmail|faumail|"
                            r"gitlab|git|fau\.tv|www\.fau\.tv|video|osm|wiki|intern|intranet|moodle|"
                            r"ilias|cris|opus4|exam|evaluation|evasys|lists|mailman|owncloud|"
                            r"faubox|vpn|wwwcip|www1|www2|www3|www4|people|home|user|"
                            r"faui\d+\w*)(\.|$)")
SKIPPED_URLS = re.compile(r"(/~|/wp-admin|/wp-json|/wp-login|/feed/?$|/tag/|/author/|/category/|"
                          r"\?share=|ical|\.ics$|/print/|/druck|replytocom|/login|/logout|"
                          r"\.(jpe?g|png|gif|svg|webp|mp4|mp3|zip|docx?|xlsx?|pptx?|tex|bib)$)",
                          re.IGNORECASE)
KEYWORDS = re.compile(r"modul|handbook|handbuch|regulation|ordnung|pruefung|prüfung|studien|"
                      r"curriculum|syllabus|master|bachelor|guide|leitfaden|studying|programme",
                      re.IGNORECASE)


def fau_host(host):
    host = host.lower()
    return (host.endswith((".fau.de", ".fau.eu", ".uni-erlangen.de")) or host in ("fau.de", "fau.eu")) \
        and not EXCLUDED_HOSTS.search(host)


def is_challenge(text):
    """The proof-of-work page of Anubis, which FAU puts in front of most of its web sites."""
    return "anubis_challenge" in text or "Making sure you&#39;re not a bot" in text or "<title>Oh noes" in text


class Crawler:
    def __init__(self):
        import requests
        self.requests = requests
        self.lock = threading.Lock()
        self.hosts = {}                      # host -> dict(queue, seen, robots, pages, pdfs, status)
        self.threads = []
        self.n_pages = self.n_pdfs = 0
        self.deadline = time.time() + 60 * CRAWL_MINUTES
        self.seq = 0
        # A crawl that was interrupted resumes: what was fetched is not fetched again.
        self.done, self.previous = set(), collections.Counter()
        if (RAW / "fetched.jsonl").exists():
            for line in open(RAW / "fetched.jsonl"):
                f = json.loads(line)
                self.done.update(u for u in (f["url"], f["final_url"]) if u)
                if f["type"] in ("html", "pdf"):
                    self.previous[(urllib.parse.urlsplit(f["url"]).hostname, f["type"])] += 1
        self.log = open(RAW / "fetched.jsonl", "a")

    # -- the queue of every host: (priority, sequence number, url, kind) --
    def add(self, url, priority, kind="page"):
        try:
            url, _ = urllib.parse.urldefrag(url)
            parts = urllib.parse.urlsplit(url)
        except ValueError:
            return
        if parts.scheme not in ("http", "https") or not fau_host(parts.hostname or ""):
            return
        if SKIPPED_URLS.search(url):
            return
        host = parts.hostname.lower()
        with self.lock:
            if url in self.done:
                return
            if host not in self.hosts:
                if sum(s["status"] != "blocked" for s in self.hosts.values()) >= MAX_HOSTS:
                    return
                self.hosts[host] = dict(queue=[], seen=set(), pages=self.previous[(host, "html")],
                                        pdfs=self.previous[(host, "pdf")], status="new", robots=None)
                thread = threading.Thread(target=self.worker, args=(host,), daemon=True)
                self.threads.append(thread)
                thread.start()
            state = self.hosts[host]
            if url in state["seen"] or state["status"] == "blocked":
                return
            state["seen"].add(url)
            self.seq += 1
            heapq.heappush(state["queue"], (priority, self.seq, url, kind))

    def get(self, session, url, limit):
        response = session.get(url, timeout=30, stream=True, allow_redirects=True)
        body = b""
        for chunk in response.iter_content(65536):
            body += chunk
            if len(body) > limit:
                response.close()
                return response, None
        return response, body

    def sitemap_urls(self, session, host, robots_text):
        """Page URLs from the sitemaps of a host (WordPress: wp-sitemap.xml or Yoast's sitemap_index.xml)."""
        maps = re.findall(r"(?im)^sitemap:\s*(\S+)", robots_text)
        maps += [f"https://{host}/wp-sitemap.xml", f"https://{host}/sitemap_index.xml"]
        found, done = [], set()
        while maps and len(done) < 40:
            url = maps.pop(0)
            if url in done:
                continue
            done.add(url)
            try:
                time.sleep(DELAY)
                response, body = self.get(session, url, 5_000_000)
            except Exception:
                continue
            if response.status_code != 200 or body is None:
                continue
            text = body.decode("utf-8", "replace")
            if is_challenge(text) or "<loc>" not in text:
                continue
            for loc in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text):
                loc = loc.replace("&amp;", "&")
                if loc.endswith(".xml"):
                    # pages and posts, not people, users, categories or tags
                    if not re.search(r"person|users|taxonom|author|category|tag", loc):
                        maps.append(loc)
                else:
                    found.append(loc)
        return found

    def worker(self, host):
        session = self.requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        state = self.hosts[host]
        try:
            time.sleep(DELAY)
            response = session.get(f"https://{host}/robots.txt", timeout=20)
            robots_text = response.text if response.status_code == 200 else ""
        except Exception:
            state["status"] = "unreachable"
            return
        if is_challenge(robots_text):
            robots_text = ""
        robots = urllib.robotparser.RobotFileParser()
        robots.parse(robots_text.splitlines())
        # Is the host behind the proof-of-work page?  Then leave it alone.
        try:
            time.sleep(DELAY)
            response = session.get(f"https://{host}/", timeout=20)
            if is_challenge(response.text):
                state["status"] = "blocked"
                print(f"  {host}: asks for a proof of work, not crawled", flush=True)
                return
        except Exception:
            state["status"] = "unreachable"
            return
        state["status"] = "crawling"
        for url in self.sitemap_urls(session, host, robots_text):
            self.add(url, 1)
        print(f"  {host}: crawling ({len(state['seen'])} URLs from its sitemaps)", flush=True)
        idle = 0
        while time.time() < self.deadline:
            with self.lock:
                item = heapq.heappop(state["queue"]) if state["queue"] else None
            if item is None:
                idle += 1
                if idle > 120:               # nothing new for two minutes: this host is done
                    break
                time.sleep(1)
                continue
            idle = 0
            priority, _, url, kind = item
            if not robots.can_fetch(USER_AGENT, url):
                continue
            is_pdf = kind == "pdf"
            with self.lock:
                if is_pdf and (state["pdfs"] >= MAX_PDFS_PER_HOST or self.n_pdfs >= MAX_PDFS):
                    continue
                if not is_pdf and (state["pages"] >= MAX_PAGES_PER_HOST or self.n_pages >= MAX_PAGES):
                    continue
            time.sleep(DELAY)
            try:
                response, body = self.get(session, url, MAX_PDF_BYTES if is_pdf else MAX_HTML_BYTES)
            except Exception as error:
                self.record(url, None, "error", 0, None, str(error)[:80])
                continue
            ctype = response.headers.get("Content-Type", "").lower()
            if response.status_code != 200 or body is None:
                self.record(url, response.url, "skip", response.status_code, None)
                continue
            if "pdf" in ctype or body[:5] == b"%PDF-":
                if body[:5] != b"%PDF-":
                    continue
                name = hashlib.sha1(response.url.encode()).hexdigest()[:16] + ".pdf"
                (RAW / name).write_bytes(body)
                with self.lock:
                    state["pdfs"] += 1
                    self.n_pdfs += 1
                self.record(url, response.url, "pdf", 200, name)
            elif "html" in ctype:
                text = body.decode(response.encoding or "utf-8", "replace")
                if is_challenge(text):
                    state["status"] = "blocked"
                    print(f"  {host}: started asking for a proof of work, stopped", flush=True)
                    return
                name = hashlib.sha1(response.url.encode()).hexdigest()[:16] + ".html.gz"
                (RAW / name).write_bytes(gzip.compress(body))
                with self.lock:
                    state["pages"] += 1
                    self.n_pages += 1
                self.record(url, response.url, "html", 200, name)
                self.links(response.url, text, priority)
        state["status"] = "done"

    def links(self, base, text, priority):
        for href in re.findall(r'href\s*=\s*["\']([^"\'#][^"\']*)["\']', text):
            try:
                url = urllib.parse.urljoin(base, href.replace("&amp;", "&"))
            except ValueError:               # malformed links exist
                continue
            if url.lower().split("?")[0].endswith(".pdf"):
                self.add(url, 0 if KEYWORDS.search(url) else 2, "pdf")
            else:
                self.add(url, max(priority, 3) + 0, "page")

    def record(self, url, final, kind, status, name, note=""):
        line = dict(url=url, final_url=final, type=kind, status=status, file=name,
                    date=datetime.datetime.now().isoformat(timespec="seconds"), note=note)
        with self.lock:
            self.log.write(json.dumps(line) + "\n")
            self.log.flush()

    def run(self):
        print(f"crawling for at most {CRAWL_MINUTES} minutes", flush=True)
        for seed in SEEDS:
            self.add(seed, 1)
        # the links of the pages fetched before an interruption
        for line in open(RAW / "fetched.jsonl"):
            f = json.loads(line)
            if f["type"] == "html":
                text = gzip.decompress((RAW / f["file"]).read_bytes()).decode("utf-8", "replace")
                self.links(f["final_url"] or f["url"], text, 3)
        last = 0
        while time.time() < self.deadline and any(t.is_alive() for t in self.threads):
            time.sleep(5)
            if time.time() - last > 120:
                last = time.time()
                print(f"  {time.strftime('%H:%M:%S')}: {self.n_pages} pages, {self.n_pdfs} PDFs, "
                      f"{len(self.hosts)} hosts", flush=True)
        time.sleep(2)
        status = {h: (s["status"], s["pages"], s["pdfs"]) for h, s in sorted(self.hosts.items())}
        (RAW / "hosts.json").write_text(json.dumps(status, indent=1))
        print(f"crawl done: {self.n_pages} pages, {self.n_pdfs} PDFs")


# ---------------------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------------------
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+|"
                   r"[\w.+-]+\s*(\(at\)|\[at\]|\s+at\s+)\s*[\w-]+(\.[\w-]+)*\s*(\.|\(dot\)|\[dot\])\s*(de|eu|com|org|net)\b",
                   re.IGNORECASE)
PHONE = re.compile(r"(\+\s?49|\(0\d{2,5}\)|\b0\d{2,5})[\s/\-()]*\d[\d\s/\-()]{4,}\d")
PHONE_LABEL = re.compile(r"(?im)^\s*(tel|phone|telefon|fax|telefax|mobil|mobile)\b\.?:?.*$")
EN_WORDS = set("the and of to in is for with are on this that be as by from at an or it can which "
               "students will their these have has not you your our we module course".split())
DE_WORDS = set("der die und das ist nicht mit für den von zu im eine auf des werden sich dem ein "
               "auch als oder wird sind bei nach zur zum einer über studierenden kann".split())


def language(text):
    words = re.findall(r"[a-zäöüß]+", text.lower())
    en = sum(w in EN_WORDS for w in words)
    de = sum(w in DE_WORDS for w in words)
    if en + de < 3:
        return None                          # too short to tell
    return "en" if en > de else "de"


BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd", "td", "th", "blockquote",
              "figcaption", "pre", "summary"]
NOISE = re.compile(r"cookie|consent|breadcrumb|skiplink|skip-link|sidebar|widget|menu|navigation|"
                   r"navbar|footer|header|social|share|banner|search|pagination|meta-|toolbar|"
                   r"language-switch|lang-switch|contact|kontakt|vcard|person|hidden|screen-reader|"
                   r"sr-only|tabs-nav|accordion-head|portal|logo|related", re.IGNORECASE)


def html_text(raw):
    """The main content of a page as paragraphs, without navigation, banners and footers."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form",
                     "button", "svg", "iframe", "select", "template", "address"]):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        names = " ".join(tag.get("class") or []) + " " + (tag.get("id") or "")
        if tag.name not in ("html", "body", "main", "article") and NOISE.search(names):
            tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    root = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body or soup
    paragraphs = []
    for block in root.find_all(BLOCK_TAGS):
        if block.find(BLOCK_TAGS):           # only the innermost blocks, so no text twice
            continue
        text = normalize(block.get_text(" ", strip=True))
        if not text:
            continue
        if block.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            paragraphs.append(("H", text))
            continue
        # a list item or paragraph that is nothing but a link is a menu, not text
        linked = sum(len(a.get_text(" ", strip=True)) for a in block.find_all("a"))
        if linked >= 0.8 * len(text):
            continue
        paragraphs.append(("P", text))
    return title, paragraphs


TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"',
                            "„": '"', "«": '"', "»": '"', "–": "-", "—": "-",
                            "−": "-", "…": "...", " ": " ", "­": "", "​": "",
                            "•": "-", "": "-", "●": "-", "▪": "-"})


def normalize(text):
    """One space between words, none before punctuation, and typographic quotes and dashes as the
    plain characters that dominate GPT-2's training text (umlauts and other letters stay)."""
    text = unicodedata.normalize("NFKC", text).translate(TYPOGRAPHY)
    text = " ".join(text.split())
    text = re.sub(r" ([.,;:!?)\]])", r"\1", text)
    return re.sub(r"([(\[]) ", r"\1", text)


def pdf_text(path):
    """Paragraphs of a PDF: pdftotext in reading order, then headers, footers, page numbers,
    tables of contents and hyphenation at line ends removed."""
    try:
        out = subprocess.run(["/usr/bin/pdftotext", "-enc", "UTF-8", "-nopgbrk", "-q", str(path), "-"],
                             capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
        pages = subprocess.run(["/usr/bin/pdftotext", "-enc", "UTF-8", "-q", str(path), "-"],
                               capture_output=True, timeout=120).stdout.decode("utf-8", "replace").split("\f")
    except Exception:
        return []
    # running headers and footers: lines that occur on more than a quarter of the pages
    counts = collections.Counter()
    for page in pages:
        counts.update({re.sub(r"\d+", "#", line.strip()) for line in page.splitlines() if line.strip()})
    repeated = {line for line, k in counts.items() if len(pages) >= 4 and k > max(2, len(pages) / 4)}
    lines = []
    for line in out.splitlines():
        s = line.strip()
        if re.sub(r"\d+", "#", s) in repeated and len(s) < 120:
            continue
        if re.fullmatch(r"(page|seite)?\s*\d+(\s*(of|von|/)\s*\d+)?", s, re.IGNORECASE):
            continue                         # page numbers
        if re.search(r"(\.\s?){4,}\s*\d+\s*$|…{2,}\s*\d+\s*$", s):
            continue                         # tables of contents with dot leaders
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n([a-zäöüß])", r"\1\2", text)          # hyphenation at line ends
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        joined = normalize(" ".join(l for l in block.splitlines() if l))
        if joined:
            paragraphs.append(("H" if len(joined) < 60 else "P", joined))
    return paragraphs


def scrub(text):
    """No e-mail addresses and no telephone numbers in the corpus."""
    text = EMAIL.sub("", text)
    return PHONE.sub("", text)


# Not documents of the university in the sense of this experiment: research papers and theses
# (the work of individuals, and most of the PDF text if kept), and pages of the login service.
PUBLICATION = re.compile(r"publication|publikation|/papers?/|thesis|arbeit|diss|preprint|/pub/|"
                         r"_\d\d_|\d\d-[A-Z]{2,}\.pdf", re.IGNORECASE)
PEOPLE = re.compile(r"/team|/staff|/people|/members|/mitarbeit|/personen|/person/|/kontakt|/contact|"
                    r"/former-members|/ehemalige|/alumni|/sprechstunde|/office-hours", re.IGNORECASE)
NEWS = re.compile(r"/news/|/\d{4}/\d{2}/\d{2}/|/aktuelles/|/neuigkeiten/|/press|/presse", re.IGNORECASE)
MAX_DOC_CHARS = 200_000                      # a few annual reports would otherwise be half the German text


def count_tokens(text):
    import gpt2
    return len(gpt2.encode(text))


def clean():
    CORPUS.mkdir(parents=True, exist_ok=True)
    fetched = [json.loads(line) for line in open(RAW / "fetched.jsonl")]
    fetched = [f for f in fetched if f["type"] in ("html", "pdf")]
    seen_urls, docs, dropped = set(), [], collections.Counter()
    for f in fetched:
        url = f["final_url"] or f["url"]
        host = urllib.parse.urlsplit(url).hostname or ""
        if url in seen_urls:
            continue
        seen_urls.add(url)
        if not fau_host(host) or SKIPPED_URLS.search(url):
            dropped["login page or excluded host"] += 1
            continue
        if PUBLICATION.search(url):
            dropped["research paper, thesis or list of publications"] += 1
            continue
        if PEOPLE.search(url):
            dropped["list of people"] += 1
            continue
        path = RAW / f["file"]
        if f["type"] == "html":
            title, paragraphs = html_text(gzip.decompress(path.read_bytes()))
            kind = "news" if NEWS.search(url) else "page"
        else:
            paragraphs, kind = pdf_text(path), "pdf"
        docs.append(dict(url=url, host=host, type=kind, date=f["date"], paragraphs=paragraphs))
    print(f"{len(docs)} documents kept of {len(seen_urls)} fetched; left out: {dict(dropped)}")

    # Boilerplate: a long paragraph that occurs in several documents is kept once; a short one
    # that occurs in more than 10 web pages ("Read more", "Contact", "Opening hours") never.
    def key(text):
        return re.sub(r"\W+", " ", text.lower()).strip()
    frequency = collections.Counter()
    for d in docs:
        if d["type"] != "pdf":
            frequency.update({key(t) for _, t in d["paragraphs"]})
    seen = set()
    for d in docs:
        kept = []
        for kind, text in d["paragraphs"]:
            text = " ".join(scrub(text).split())
            k = key(text)
            if len(k) < 2:
                continue
            if len(k) < 60:
                if d["type"] != "pdf" and frequency[k] > 10:
                    continue
            elif k in seen:
                continue
            seen.add(k)
            kept.append(text)
        d["paragraphs"] = kept

    # Language per paragraph; the document's language is that of most of its characters, and a
    # document keeps only the paragraphs of its language (and the short ones that have none).
    for d in docs:
        langs = [language(text) for text in d["paragraphs"]]
        chars = collections.Counter()
        for text, lang in zip(d["paragraphs"], langs):
            if lang:
                chars[lang] += len(text)
        d["lang"] = chars.most_common(1)[0][0] if chars else None
        d["text"] = "\n".join(t for t, lang in zip(d["paragraphs"], langs) if lang in (None, d["lang"]))
        d["text"] = d["text"][:MAX_DOC_CHARS].strip()
        del d["paragraphs"]

    docs = [d for d in docs if d["lang"] and len(d["text"]) >= 300]
    with multiprocessing.Pool(8) as pool:             # the tokenizer is plain Python: 8 processes
        tokens = pool.map(count_tokens, [d["text"] for d in docs], chunksize=8)
    for d, k in zip(docs, tokens):
        d["split"] = "val" if int(hashlib.sha1(d["url"].encode()).hexdigest(), 16) % 20 == 0 else "train"
        d["chars"] = len(d["text"])
        d["tokens"] = k
        d["words"] = len(d["text"].split())

    with open(CORPUS / "manifest.jsonl", "w") as manifest:
        for d in docs:
            manifest.write(json.dumps({k: d[k] for k in ("url", "host", "type", "lang", "split", "chars",
                                                         "words", "tokens", "date")}) + "\n")
    for lang in ("en", "de"):
        for split in ("train", "val"):
            texts = [d["text"] for d in docs if d["lang"] == lang and d["split"] == split]
            (CORPUS / f"{lang}_{split}.txt").write_text(SEPARATOR.join(texts) + "\n")
    report()


SEPARATOR = "\n<|endoftext|>\n"                # between documents; read as GPT-2's token 50256


def report():
    """The size of the corpus, per language, source type and split; and a few random samples."""
    docs = [json.loads(line) for line in open(CORPUS / "manifest.jsonl")]
    print("\ncorpus (after cleaning):")
    print(f"  {'lang':4} {'type':4} {'split':5} {'docs':>6} {'chars':>11} {'words':>10} {'tokens':>10} {'tok/word':>8}")
    groups = collections.defaultdict(list)
    for d in docs:
        groups[(d["lang"], d["type"], d["split"])].append(d)
        groups[(d["lang"], "all", d["split"])].append(d)
    stats = {}
    for key in sorted(groups):
        g = groups[key]
        chars, words, tokens = (sum(d[k] for d in g) for k in ("chars", "words", "tokens"))
        stats["/".join(key)] = dict(docs=len(g), chars=chars, words=words, tokens=tokens)
        print(f"  {key[0]:4} {key[1]:4} {key[2]:5} {len(g):6d} {chars:11,d} {words:10,d} {tokens:10,d} {tokens / words:8.2f}")
    hosts = collections.Counter(d["host"] for d in docs if d["lang"] == "en")
    print("  English documents per host:", dict(hosts.most_common()))
    hosts = collections.Counter(d["host"] for d in docs if d["lang"] == "de")
    print("  German documents per host:", dict(hosts.most_common()))
    (CORPUS / "stats.json").write_text(json.dumps(stats, indent=1))
    if (RAW / "hosts.json").exists():
        status = json.loads((RAW / "hosts.json").read_text())
        print("  hosts crawled:", sorted(h for h, s in status.items() if s[1] + s[2] > 0))
        print("  hosts that ask for a proof of work (not crawled):", sorted(h for h, s in status.items() if s[0] == "blocked"))


if __name__ == "__main__":
    if "--crawl" in sys.argv:                                 # never on its own: make_figures.py must not start a crawl
        RAW.mkdir(parents=True, exist_ok=True)
        Crawler().run()
    if not (RAW / "fetched.jsonl").exists():
        print("no corpus in data/fau/ yet: run this script with --crawl (it fetches public pages of FAU for up "
              "to CRAWL_MINUTES, default 40), then part4_finetune.py --train")
    elif "--clean" in sys.argv or "--crawl" in sys.argv or not (CORPUS / "manifest.jsonl").exists():
        clean()
    else:
        report()
