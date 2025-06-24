from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./products.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    product_id = Column(String(30), primary_key=True, index=True, unique=True)
    name = Column(String, nullable=False)
    product_url = Column(String, nullable=False, unique=True)
    price = Column(Float, nullable=False)
    currency = Column(String(10))
    discount_percentage = Column(Integer)
    delivery_type = Column(String(255))
    image_url = Column(String)
    category = Column(String(100))
    colour = Column(String(100))
    material = Column(String(100))
    pattern = Column(String(100))
    occasion = Column(String(100))
    garment_type = Column(String(100))
    shoe_type = Column(String(100))
    bag_type = Column(String(100))
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def populate_sample_data():
    # Optionally populate with sample data from data/products.json
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_all_products(db, search=None, categories=None, brand=None, min_price=None, max_price=None):
    query = db.query(Product)
    if search:
        query = query.filter(Product.name.ilike(f"%{search}%"))
    if categories:
        query = query.filter(Product.category.in_(categories))
    if brand:
        query = query.filter(Product.name.ilike(f"%{brand}%"))
    if min_price is not None:
        query = query.filter(Product.price >= min_price)
    if max_price is not None:
        query = query.filter(Product.price <= max_price)
    return query.all()

def get_filter_options(db):
    categories = db.query(Product.category).distinct().all()
    return {"categories": [c[0] for c in categories if c[0]]} 