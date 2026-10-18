import os
import sqlite3
import zipfile
import pandas as pd

# Paths and configuration
CSV_PATH = "gdp_1960_2020.csv"
ZIP_PATH = "archive.zip"
DB_PATH = "world_gdp.db"
TABLE_NAME = "world_gdp"


def extract_data_if_needed() -> str:
    """Ensure gdp_1960_2020.csv is available, extracting it from archive.zip if necessary."""
    if os.path.exists(CSV_PATH):
        print(f"Found existing CSV dataset: '{CSV_PATH}'")
        return CSV_PATH

    if os.path.exists(ZIP_PATH):
        print(f"Extracting '{CSV_PATH}' from '{ZIP_PATH}'...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extract(CSV_PATH)
        print(f"Extracted '{CSV_PATH}' successfully.")
        return CSV_PATH

    raise FileNotFoundError(
        f"Neither dataset file '{CSV_PATH}' nor archive '{ZIP_PATH}' was found in {os.getcwd()}."
    )


def load_and_preprocess_data() -> pd.DataFrame:
    """Load and clean the World Countries GDP dataset."""
    csv_file = extract_data_if_needed()
    print(f"Loading data from '{csv_file}'...")
    df = pd.read_csv(csv_file)

    # Standardize column headers
    df.columns = [col.strip().lower() for col in df.columns]

    # Clean text columns
    for col in ["country", "state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Add 'continent' column as an alias of 'state' for semantic clarity
    if "state" in df.columns and "continent" not in df.columns:
        df["continent"] = df["state"]

    # Ensure numeric columns are properly typed
    numeric_int_cols = ["year", "rank"]
    for col in numeric_int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    numeric_float_cols = ["gdp", "gdp_percent"]
    for col in numeric_float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any potential exact duplicate rows
    initial_count = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df):,} duplicate rows.")

    print(f"Preprocessed {len(df):,} records across {len(df.columns)} columns.")
    return df


def store_in_sqlite(df: pd.DataFrame, db_path: str = DB_PATH, table_name: str = TABLE_NAME):
    """Store the cleaned DataFrame in an SQLite database with analytical indexes."""
    # Validate existing database integrity if present
    if os.path.exists(db_path):
        try:
            with open(db_path, "rb") as f:
                header = f.read(16)
            if not header.startswith(b"SQLite format 3"):
                print(f"Overwriting invalid/corrupted database file '{db_path}'...")
                os.remove(db_path)
        except Exception:
            pass

    print(f"\nConnecting to SQLite database: '{db_path}'...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Write DataFrame to SQLite
    print(f"Writing {len(df):,} records into table '{table_name}'...")
    df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)

    # Create analytical indexes for fast query performance
    print("Creating database indexes for optimized querying...")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_year ON {table_name}(year);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_country ON {table_name}(country);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_year_country ON {table_name}(year, country);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_continent ON {table_name}(continent);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_rank_year ON {table_name}(year, rank);")
    conn.commit()

    # Database Verification & Analytics Summary
    print("\n" + "=" * 55)
    print("       DATABASE VERIFICATION & ANALYTICS SUMMARY")
    print("=" * 55)

    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    total_records = cursor.fetchone()[0]
    print(f"Total Records Inserted : {total_records:,}")

    cursor.execute(f"SELECT COUNT(DISTINCT country) FROM {table_name};")
    total_countries = cursor.fetchone()[0]
    print(f"Distinct Countries     : {total_countries}")

    cursor.execute(f"SELECT MIN(year), MAX(year) FROM {table_name};")
    min_year, max_year = cursor.fetchone()
    print(f"Year Range             : {min_year} - {max_year}")

    # Top 5 economies in the latest available year
    print(f"\nTop 5 Economies in {max_year}:")
    cursor.execute(f"""
        SELECT rank, country, continent, gdp, ROUND(gdp_percent * 100, 2)
        FROM {table_name}
        WHERE year = ?
        ORDER BY rank ASC
        LIMIT 5;
    """, (max_year,))
    for rank, country, continent, gdp, share in cursor.fetchall():
        print(f"  #{rank:<2} {country:<20} ({continent:<7}) : ${gdp:,.0f} ({share:.2f}% of world)")

    # Continental breakdown for latest year
    print(f"\nContinental Breakdown in {max_year}:")
    cursor.execute(f"""
        SELECT 
            continent,
            COUNT(*) as country_count,
            SUM(gdp) as total_gdp,
            ROUND(SUM(gdp_percent) * 100, 2) as world_share_pct
        FROM {table_name}
        WHERE year = ?
        GROUP BY continent
        ORDER BY total_gdp DESC;
    """, (max_year,))
    for continent, count, total_gdp, share in cursor.fetchall():
        print(f"  {continent:<10} | {count:2d} countries | Total GDP: ${total_gdp:,.0f} ({share:.2f}%)")

    conn.close()
    print("=" * 55)
    print(f"Data successfully pushed to SQLite database: '{db_path}'!\n")


if __name__ == "__main__":
    df_clean = load_and_preprocess_data()
    store_in_sqlite(df_clean)

