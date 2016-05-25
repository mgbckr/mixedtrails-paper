import pymysql as db
db.install_as_MySQLdb()

import datetime

def write(scope, details, ks, evidences, con, start_time = None, end_time = None, replace=False, table="results"):
    
    if start_time is None:
        start_time = datetime.datetime.now()
        
    if end_time is None:
        end_time = datetime.datetime.now()

    with con.cursor() as cursor:
        if replace:
            sql = "DELETE FROM {} WHERE scope=%s and details=%s".format(table)
            cursor.execute(sql, (scope, details))

        sql = "INSERT INTO {} (scope, details, ks, evidences, start_time, end_time) \
            VALUES (%s, %s, %s, %s, %s, %s)".format(table)

        cursor.execute(sql, (
                scope, \
                details, \
                ",".join([str(k) for k in ks]), \
                ",".join([str(e) for e in evidences]), \
                start_time.strftime('%Y-%m-%d %H:%M:%S'), \
                end_time.strftime('%Y-%m-%d %H:%M:%S')))

        # connection is not autocommit by default. So you must commit to save
        # your changes.
        con.commit()

    
    