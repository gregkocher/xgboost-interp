"""
Utility for tracking analysis successes and failures during example runs.
"""


class AnalysisTracker:
    """Tracks which analyses succeeded or failed, and prints a summary at the end."""

    def __init__(self):
        self.results = []  # list of (name, success, error_msg)

    def success(self, name: str) -> None:
        """Record a successful analysis."""
        self.results.append((name, True, None))

    def failure(self, name: str, error) -> None:
        """Record a failed analysis."""
        self.results.append((name, False, str(error)))

    def print_summary(self) -> None:
        """Print a summary of all tracked analyses, highlighting failures."""
        if not self.results:
            return

        failures = [(n, e) for n, ok, e in self.results if not ok]
        successes = [n for n, ok, _ in self.results if ok]

        print("\n" + "=" * 70)
        print("ANALYSIS TRACKER SUMMARY")
        print("=" * 70)
        print(f"  Total:     {len(self.results)}")
        print(f"  Succeeded: {len(successes)}")
        print(f"  Failed:    {len(failures)}")

        if failures:
            print(f"\nFailed analyses:")
            for name, err in failures:
                print(f"  - {name}: {err}")
        else:
            print("\nAll analyses completed successfully.")
        print("=" * 70)
