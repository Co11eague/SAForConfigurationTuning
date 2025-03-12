import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


def main():
    MAXIMISATION = False
    system = "brotli"
    rs_df = pd.read_csv('data/RS.csv')
    isa_df = pd.read_csv('data/ISA.csv')

    isa_df["difference"] = isa_df[system] - rs_df[system]


    stat, p = shapiro(isa_df["difference"])

    print(f"Shapiro-Wilk Test Statistic: {stat}, p-value: {p}")

    if p > 0.05:
        print("We will be using paired t-test to analyze it.")
        t_stat, p_value = ttest_rel(isa_df[system], rs_df[system])

        print(f"Paired t-test: t={t_stat}, p={p_value}")
        if p_value < 0.05:
            if not MAXIMISATION:
                    if t_stat < 0:
                        print("ISA performs significantly better than RS")
                    else:
                        print("RS performs significantly better than ISA")
            else:
                if t_stat > 0:
                    print("ISA performs significantly better than RS")
                else:
                    print("RS performs significantly better than ISA")
        else:
            print("No significant difference")

    else:
        w_stat, p_value = wilcoxon(isa_df[system], rs_df[system])

        print(f"Wilcoxon Test Statistic: {w_stat}, p-value: {p_value}")
        if p_value < 0.05:
            median_difference = isa_df["difference"].median()
            print(f"Median difference: {median_difference}")

            if not MAXIMISATION:
                    if median_difference < 0:
                        print("ISA performs significantly better than RS")
                    else:
                        print("RS performs significantly better than ISA")
            else:
                    if median_difference > 0:
                        print("ISA performs significantly better than RS")
                    else:
                        print("RS performs significantly better than ISA")
        else:
            print("No significant difference")

if __name__ == "__main__":
    main()