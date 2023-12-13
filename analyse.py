import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = "new_churn_data.csv" # "new_spambase_data.csv" # "adult_data.csv"

          # "mushroom_data.csv"
    # "soinn_demo_train.mat"
# "new_hribm_data.csv"
# "bank_data.csv"
filename = "outputs/" + dataset.split(".")[0] + "_dsos_output.csv"
df = pd.read_csv(filename)

data = df[df["resol"]==0.1]

# sns.boxplot(x="alpha", y="meanerror", data=data)
# plt.show()
#
# sns.boxplot(x="alpha", y="swarmsize", data=data)
# plt.show()

sns.boxplot(x="k", y="testacc", data=data)
plt.show()


