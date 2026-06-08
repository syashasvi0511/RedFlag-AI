import os

os.makedirs("/root/.kaggle", exist_ok=True)

# Paste your token here
token = "YOUR_KAGGLE_TOKEN"
username = "syashasvi0511"  # ← replace this

kaggle_json = f'{{"username":"{username}","key":"{token}"}}'

with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write(kaggle_json)

os.chmod("/root/.kaggle/kaggle.json", 0o600)
print("Credentials saved!")
