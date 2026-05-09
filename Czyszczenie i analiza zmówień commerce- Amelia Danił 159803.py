import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta


# CZĘŚĆ 0 — GENEROWANIE BRUDNYCH DANYCH

np.random.seed(42)

n = 500

klienci = [
    "Anna Kowalska", "  Jan Nowak", "Anna Kowalska", "PIOTR WIŚNIEWSKI",
    "katarzyna lewandowska", "Tomasz Zieliński ", "Marta Wójcik",
    "anna kowalska ", "Krzysztof Kamiński", " Magdalena Dąbrowska"
]

produkty = [
    "Laptop", "Mysz", "Klawiatura", "Monitor", "laptop", "MYSZ",
    "Słuchawki", "Pendrive", "monitor", "Webcam"
]

kategorie = [
    "Elektronika", "elektronika", "ELEKTRONIKA", "Akcesoria",
    "akcesoria", "Akcesoria "
]

miasta = [
    "Warszawa", "Kraków", "warszawa", "Gdańsk", "WROCŁAW",
    "Poznań", "Łódź ", " Warszawa", "kraków"
]

start_date = datetime(2025, 1, 1)

daty_iso = [
    (start_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
    for d in np.random.randint(0, 300, n // 2)
]

daty_pl = [
    (start_date + timedelta(days=int(d))).strftime("%d.%m.%Y")
    for d in np.random.randint(0, 300, n // 2)
]

daty = daty_iso + daty_pl
np.random.shuffle(daty)

df = pd.DataFrame({
    "order_id": range(1001, 1001 + n),
    "klient": np.random.choice(klienci, n),
    "produkt": np.random.choice(produkty, n),
    "kategoria": np.random.choice(kategorie, n),
    "miasto": np.random.choice(miasta, n),
    "ilosc": np.random.choice(
        [1, 2, 3, 5, -1, 0],
        n,
        p=[0.5, 0.2, 0.15, 0.1, 0.025, 0.025]
    ),
    "cena_jednostkowa": np.random.choice(
        ["199.99", "299,99", "1 499.00", "89.50", "2999", "399.00 zł", None, "abc"],
        n
    ),
    "data_zamowienia": daty,
    "email": np.random.choice(
        [
            "anna@gmail.com", "JAN@WP.PL", "piotr.w@onet", "marta@gmail.com",
            "tomasz@interia.pl", None, "krzysztof.k@gmail.com", "brak"
        ],
        n
    )
})

for col in ["miasto", "kategoria", "data_zamowienia"]:
    df.loc[df.sample(frac=0.05, random_state=1).index, col] = np.nan

df = pd.concat([df, df.sample(20, random_state=2)], ignore_index=True)

df.to_csv("zamowienia_messy.csv", index=False)

print("Wygenerowano plik zamowienia_messy.csv")
print("Liczba wierszy:", len(df))


# CZĘŚĆ 1 — EKSPLORACJA I IDENTYFIKACJA PROBLEMÓW

df = pd.read_csv("zamowienia_messy.csv")

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- INFO ---")
print(df.info())

print("\n--- DESCRIBE ---")
print(df.describe(include="all"))

print("\n--- BRAKI DANYCH ---")
print(df.isnull().sum())

print("\n--- VALUE COUNTS: klient ---")
print(df["klient"].value_counts())

print("\n--- VALUE COUNTS: produkt ---")
print(df["produkt"].value_counts())

print("\n--- VALUE COUNTS: kategoria ---")
print(df["kategoria"].value_counts())

print("\n--- VALUE COUNTS: miasto ---")
print(df["miasto"].value_counts())

# Zidentyfikowane problemy z jakością danych:
# 1. W danych znajdują się duplikaty wierszy.
# 2. W kolumnach tekstowych występują zbędne spacje na początku lub końcu tekstu.
# 3. Te same wartości zapisane są różną wielkością liter, np. Laptop/laptop, MYSZ/Mysz.
# 4. Kolumna cena_jednostkowa zawiera różne formaty, np. przecinek zamiast kropki, spacje, "zł", wartości "abc" oraz braki.
# 5. Kolumna data_zamowienia zawiera dwa różne formaty dat oraz braki danych.
# 6. W kolumnie ilosc występują wartości błędne: 0 oraz -1.
# 7. W kolumnie email znajdują się wartości niepoprawne, np. "brak" albo adres bez pełnej domeny.
# 8. W kolumnach miasto i kategoria występują braki danych.


# CZĘŚĆ 2 — CZYSZCZENIE DANYCH

print("\n--- CZYSZCZENIE DANYCH ---")

df = df.drop_duplicates()

df["klient"] = df["klient"].str.strip().str.title()
df["produkt"] = df["produkt"].str.strip().str.title()
df["kategoria"] = df["kategoria"].str.strip().str.lower()
df["miasto"] = df["miasto"].str.strip().str.title()

df["data_zamowienia"] = pd.to_datetime(
    df["data_zamowienia"],
    errors="coerce",
    dayfirst=True
)

df["cena_jednostkowa"] = (
    df["cena_jednostkowa"]
    .astype(str)
    .str.replace("zł", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.replace(",", ".", regex=False)
)

df["cena_jednostkowa"] = pd.to_numeric(
    df["cena_jednostkowa"],
    errors="coerce"
)

df["miasto"] = df["miasto"].fillna("Unknown")
df["kategoria"] = df["kategoria"].fillna("unknown")
df["email"] = df["email"].fillna("brak_emaila")

df = df.dropna(subset=["cena_jednostkowa", "data_zamowienia"])

df = df[df["ilosc"] > 0]

print("Liczba wierszy po czyszczeniu:", len(df))
print(df.isnull().sum())


# CZĘŚĆ 3 — TRANSFORMACJE

df["wartosc_zamowienia"] = df["ilosc"] * df["cena_jednostkowa"]

df["rok"] = df["data_zamowienia"].dt.year
df["miesiac"] = df["data_zamowienia"].dt.month
df["nazwa_dnia"] = df["data_zamowienia"].dt.day_name()

wzorzec_email = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

df["email_poprawny"] = df["email"].apply(
    lambda email: bool(re.match(wzorzec_email, str(email)))
)


# CZĘŚĆ 4 — ANALIZA SQL-STYLE

print("\n--- Łączna wartość zamówień w każdym miesiącu ---")
wartosc_miesieczna = df.groupby("miesiac")["wartosc_zamowienia"].sum()
print(wartosc_miesieczna)

print("\n--- Top 5 klientów pod względem łącznej wartości zamówień ---")
top_klienci = (
    df.groupby("klient")["wartosc_zamowienia"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top_klienci)

print("\n--- Średnia wartość zamówienia w każdej kategorii ---")
srednia_kategoria = df.groupby("kategoria")["wartosc_zamowienia"].mean()
print(srednia_kategoria)


# CZĘŚĆ 5 — WIZUALIZACJA

wartosc_miesieczna.plot(kind="bar")
plt.title("Łączna wartość zamówień w każdym miesiącu")
plt.xlabel("Miesiąc")
plt.ylabel("Łączna wartość zamówień")
plt.tight_layout()
plt.show()


# CZĘŚĆ 6 — ZAPIS

df.to_csv("zamowienia_clean.csv", index=False)

print("\nZapisano oczyszczony plik jako zamowienia_clean.csv")