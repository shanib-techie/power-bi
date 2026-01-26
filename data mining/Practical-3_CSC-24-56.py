import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Funtion to run Apriori algorithm and display results
def run_apriori(df, min_sup, min_conf):
    print(f"\nRunning Apriori with Support = {min_sup}, Confidence = {min_conf}\n")

    frequent_items = apriori(df, min_support=min_sup, use_colnames=True)

    if frequent_items.empty:
        print("No frequent itemsets found. Try lowering support.")
        return

    print("Frequent Itemsets:")
    print(frequent_items)

    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_conf)

    if rules.empty:
        print("No association rules found. Try lowering confidence.")
        return

    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])



# LOAD DATASET (Groceries dataset)
input_file_path = "C:\\Users\\anshu\\OneDrive\\Desktop\\College\\DSE (Discipline Specific Elective Courses)\\" \
    "Semester 3 - Data Mining I\\Practicals\\Datasets\\Groceries_dataset.csv"
groceries = pd.read_csv(input_file_path)

# Convert dataset: item per row → list per transaction
basket = (groceries.groupby(['Member_number','Date'])['itemDescription']
          .apply(list)
          .reset_index())

# Convert to one-hot encoded format
te = TransactionEncoder()
te_data = te.fit(basket['itemDescription']).transform(basket['itemDescription'])
df = pd.DataFrame(te_data, columns=te.columns_)

# Ask user for min support and confidence
min_sup = float(input("Enter minimum support (e.g., 0.50): "))
min_conf = float(input("Enter minimum confidence (e.g., 0.75): "))

# Run Apriori
run_apriori(df, min_sup, min_conf)
