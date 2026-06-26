import os
from dotenv import load_dotenv
from snowflake import connector
import pandas as pd

load_dotenv()

def get_connection():
    return connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        authenticator=os.getenv("SNOWFLAKE_AUTHENTICATOR"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE")
    )

def run_query(query):
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    query = """
    SELECT *
    FROM FREIGHT.PETROLOGISTICS.CARGO_ON_WATER
    LIMIT 10
    """
    try:
        df = run_query(query)
        print(df.head())
    except Exception as e:
        print("Query failed")
        print(e)