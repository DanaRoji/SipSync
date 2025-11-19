import sqlite3
from datetime import datetime, timedelta, timezone

DB = "sip_sync.db"

def create_tables(conn):
    c = conn.cursor()
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS venue (
        venue_id INTEGER PRIMARY KEY,
        city TEXT,
        opening_hours TEXT,
        approximate_capacity INTEGER,
        number_of_screens INTEGER
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS drink_master (
        drink_id INTEGER PRIMARY KEY,
        base_price REAL,
        name TEXT,
        category TEXT,
        is_active INTEGER
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS pricing_rules (
        rule_id INTEGER PRIMARY KEY,
        venue_id INTEGER,
        alpha REAL,
        beta REAL,
        gama REAL,
        max_up_pct REAL,
        max_down_pct REAL,
        min_step REAL,
        FOREIGN KEY(venue_id) REFERENCES venue(venue_id)
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS event_context (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        venue_id INTEGER,
        tick TIMESTAMP,
        event_factor REAL,
        noise_level REAL,
        description TEXT,
        FOREIGN KEY(venue_id) REFERENCES venue(venue_id)
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        venue_id INTEGER,
        drink_id INTEGER,
        stock_units INTEGER,
        reorder_point INTEGER,
        stock_target INTEGER,
        last_update TIMESTAMP,
        FOREIGN KEY(venue_id) REFERENCES venue(venue_id),
        FOREIGN KEY(drink_id) REFERENCES drink_master(drink_id)
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS pricing_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tick TIMESTAMP,
        venue_id INTEGER,
        drink_id INTEGER,
        current_price REAL,
        D REAL,
        E REAL,
        I REAL,
        FOREIGN KEY(venue_id) REFERENCES venue(venue_id),
        FOREIGN KEY(drink_id) REFERENCES drink_master(drink_id)
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        venue_id INTEGER,
        drink_id INTEGER,
        customer_id INTEGER,
        timestamp TIMESTAMP,
        qty INTEGER,
        FOREIGN KEY(venue_id) REFERENCES venue(venue_id),
        FOREIGN KEY(drink_id) REFERENCES drink_master(drink_id)
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS debug_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tick TIMESTAMP,
        venue_id INTEGER,
        drink_id INTEGER,
        last_price REAL,
        raw_price REAL,
        capped_price REAL,
        rounded_price REAL,
        final_price REAL,
        reason TEXT
    )""")
    
    conn.commit()

def seed_data(conn):
    c = conn.cursor()
    
    c.execute("""
        INSERT OR REPLACE INTO venue(venue_id, city, opening_hours, approximate_capacity, number_of_screens) 
        VALUES (?,?,?,?,?)
    """, (1, "Madrid", "18:00-03:00", 200, 6))

    drinks = [
        (1, 8.0, "Mojito", "cocktail", 1),
        (2, 6.0, "Gin Tonic", "cocktail", 1),
        (3, 4.0, "Beer", "beer", 1),
        (4, 5.0, "Vodka Shot", "shot", 1),
    ]
    c.executemany("""
        INSERT OR REPLACE INTO drink_master(drink_id, base_price, name, category, is_active) 
        VALUES (?,?,?,?,?)
    """, drinks)

    c.execute("""
        INSERT OR REPLACE INTO pricing_rules(rule_id, venue_id, alpha, beta, gama, max_up_pct, max_down_pct, min_step)
        VALUES (?,?,?,?,?,?,?,?)
    """, (1, 1, 0.15, 0.05, 0.03, 0.60, 0.50, 0.05))
    
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    inventory = [
        (1, 1, 100, 20, 120, now),
        (1, 2, 120, 30, 150, now),
        (1, 3, 200, 50, 300, now),
        (1, 4, 80, 10, 100, now),
    ]
    for venue_id, drink_id, stock, reorder, target, last in inventory:
        c.execute("""
            INSERT INTO inventory(venue_id, drink_id, stock_units, reorder_point, stock_target, last_update)
            VALUES (?,?,?,?,?,?)
        """, (venue_id, drink_id, stock, reorder, target, last))
    
    # Customers
    customers = [
        (1, "Dana"),
        (2, "Alex"),
        (3, "Jamie"),
        (4, "Morgan"),
    ]
    c.executemany("INSERT OR REPLACE INTO customers(customer_id, name) VALUES (?,?)", customers)
    
    tick = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO event_context(venue_id, tick, event_factor, noise_level, description) 
        VALUES (?,?,?,?,?)
    """, (1, tick, 0.1, 0.2, "Quiet opening night"))
    
    conn.commit()
def main():
    conn = sqlite3.connect(DB)
    create_tables(conn)
    seed_data(conn)
    conn.close()
    print(f"Data base created: {DB}")

if __name__ == "__main__":
    main()