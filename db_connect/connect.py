from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2

"""
Test database was created with the following:

CREATE TABLE test(
    test_id INT PRIMARY KEY,
    test_title VARCHAR(160)
);

INSERT INTO test (test_id, test_title) VALUES (1, N'Test Title');
"""


def sqlalchemy_test_connection(user,
                               password,
                               host,
                               database):
    """
    Connect to postgres db with sqlalchemy and execute test query on test table
    """
    
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}/{database}')
    with engine.connect() as connection:
        result = connection.execute(text(r"SELECT * FROM test"))
        print('------------ SQLALCHEMY RESULT ------------')
        for row in result:
            print(row)


def psycopg2_test_connection(user,
                             password,
                             host,
                             database):
    """
    Connect to postgres db with psycopg2 and execute test query on test table
    """
    conn = psycopg2.connect(
        user = user,
        password = password,
        host = host,
        database = database
    )
    
    cur = conn.cursor()
    cur.execute(r"""SELECT * FROM test""")
    res = cur.fetchone()
    print('------------ PSYCOPG2 RESULT ------------')
    print(res)
    cur.close()
    
    
if __name__ == "__main__":
    
    host = "localhost"
    database = "ignize_test"
    user = "postgres"
    password = "Rubikova:Kocka98"
    port = '5432'
    
    sqlalchemy_test_connection(host = "localhost",
                               database = "ignize_test",
                               user = "postgres",
                               password = "Rubikova:Kocka98")
    
    psycopg2_test_connection(host = "localhost",
                             database = "ignize_test",
                             user = "postgres",
                             password = "Rubikova:Kocka98")

