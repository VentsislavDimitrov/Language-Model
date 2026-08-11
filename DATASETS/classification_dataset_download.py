import urllib.request, zipfile, os, pandas
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    with urllib.request.urlopen(url) as response: #A
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref: #B
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path) #C
    print(f"File downloaded and saved as {data_file_path}")




dataframe = pandas.read_csv("SMSSpamCollection.tsv", sep="\t", header = None, names=["Label", "Text"])


def balanced_dataset(dataframe):
    amount_of_spams = dataframe[dataframe["Label"] == "spam"].shape[0]
    balance_ham = dataframe[dataframe["Label"] == "ham"].sample(amount_of_spams, random_state = 123)
    return pandas.concat([balance_ham, dataframe[dataframe["Label"] == "spam"]])

def split(dataframe, train_amount, validation_amount):
    dataframe = dataframe.sample(frac=1, random_state = 123).reset_index(drop=True)
    train = int(len(dataframe) * train_amount)
    validation = train + int(len(dataframe) * validation_amount)

    train_dataframe = dataframe[:train]
    validation_dataframe = dataframe[train:validation]
    test_dataframe = dataframe[validation:]

    return train_dataframe, validation_dataframe, test_dataframe


balanced = balanced_dataset(dataframe)
balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})


train_dataframe, validation_dataframe, test_dataframe = split(balanced,0.6, 0.3)

train_dataframe.to_csv("train_dataset.csv", index=None)
validation_dataframe.to_csv("validation_dataset.csv", index=None)
test_dataframe.to_csv("test_dataset.csv", index=None)