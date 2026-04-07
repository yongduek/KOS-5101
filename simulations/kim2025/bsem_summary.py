import glob
import cmdstanpy

csv_files = sorted(glob.glob('mediation_bsem-*.csv'))
if not csv_files:
    print('No CSV files found. Run simulation_bsem.py first.')
else:
    print(f"Found {len(csv_files)} chain CSV files.")
    fit = cmdstanpy.from_csv(csv_files)
    summary = fit.summary()

    key = ['a', 'b', 'cp', 'indirect_effect', 'total_effect', 'sigma_m', 'sigma_y']
    print(f"\n  {'Param':<22} {'Mean':>8} {'SD':>7} {'5%':>8} {'95%':>8} {'Rhat':>7}")
    print("  " + "-" * 60)
    for p in key:
        if p in summary.index:
            r = summary.loc[p]
            print(f"  {p:<22} {r['Mean']:>8.3f} {r['StdDev']:>7.3f} {r['5%']:>8.3f} {r['95%']:>8.3f} {r['R_hat']:>7.3f}")
