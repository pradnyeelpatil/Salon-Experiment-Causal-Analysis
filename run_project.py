from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from generate_data import generate_all
from analysis import run_analysis
from make_reports import create_reports

def main():
    print("\n=== SALON EXPERIMENT & CAUSAL ANALYSIS ===")
    print("1/3 Generating reproducible synthetic data...")
    generate_all(ROOT)
    print("\n2/3 Cleaning data and running statistical analysis...")
    results = run_analysis(ROOT)
    print("\n3/3 Creating Word deliverables...")
    create_reports(ROOT, results)
    print("\nCOMPLETE. Check outputs/, visualizations/ and reports/.")
    print("Reminder: all generated records are synthetic, not real salon data.")

if __name__ == "__main__":
    main()
