-- checked with Lean 4.35.0 (core library, no Mathlib)
def sumOdd : Nat -> Nat               -- 1 + 3 + ... + (2n - 1)
  | 0     => 0
  | n + 1 => sumOdd n + (2 * n + 1)

theorem sumOdd_eq (n : Nat) : sumOdd n = n * n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    have step : (k + 1) * (k + 1) = k * k + 2 * k := by
      rw [Nat.add_mul, Nat.mul_add]; omega
    rw [sumOdd, ih, step]
