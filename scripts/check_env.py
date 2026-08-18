import os
import sys

def run_environment_check():
    print("=" * 75)
    print("HYBRID RESTAURANT RECOMMENDATION SYSTEM -- ENVIRONMENT SANITY CHECK")
    print("=" * 75)
    
    # 1. Check Python Version
    py_version = sys.version.split()[0]
    print(f"[*] Python Version: {py_version} (Executable: {sys.executable})")
    
    # 2. Check Package Imports and Versions
    packages = [
        ("fastapi", "FastAPI Web Framework"),
        ("uvicorn", "ASGI Server"),
        ("pydantic", "Data Validation & Schema Modeling"),
        ("pydantic_settings", "Settings Management"),
        ("sqlalchemy", "SQLAlchemy 2.0 ORM"),
        ("asyncpg", "Async PostgreSQL Driver"),
        ("alembic", "Database Migrations"),
        ("scipy", "Scientific Computing"),
        ("sklearn", "Scikit-Learn ML Suite"),
        ("surprise", "Scikit-Surprise Collaborative Filtering"),
        ("pandas", "Dataframe Operations"),
        ("numpy", "Numerical Computing"),
        ("joblib", "Model Serialization"),
        ("streamlit", "Frontend Dashboard"),
        ("pytest", "Testing Framework"),
        ("pytest_asyncio", "Async Testing Plugin"),
        ("httpx", "Async HTTP Client for Testing"),
        ("requests", "HTTP Client"),
    ]
    
    print("\n[*] Checking Dependency Package Versions:")
    all_packages_ok = True
    for pkg_name, desc in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  [OK] {pkg_name:<20} : v{ver:<12} ({desc})")
        except ImportError as e:
            print(f"  [FAIL] {pkg_name:<20} : Missing ({e})")
            all_packages_ok = False
            
    # 3. Check Directory Structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n[*] Project Base Directory: {base_dir}")
    
    required_dirs = [
        "app", "app/api", "app/api/v1", "app/api/v1/endpoints",
        "app/database", "app/models", "app/schemas", "app/services",
        "ml", "ml/data", "ml/preprocessing", "ml/content_based",
        "ml/collaborative", "ml/location", "ml/popularity",
        "ml/hybrid", "ml/evaluation", "ml/training",
        "saved_models", "data/raw", "data/processed",
        "frontend", "frontend/components", "frontend/utils",
        "scripts", "tests"
    ]
    
    missing_dirs = []
    for rd in required_dirs:
        full_path = os.path.join(base_dir, rd)
        if not os.path.isdir(full_path):
            missing_dirs.append(rd)
            
    if not missing_dirs:
        print(f"  [OK] All {len(required_dirs)} required project directories verified.")
    else:
        print(f"  [FAIL] Missing directories: {missing_dirs}")
        
    # 4. Check Pydantic Settings & Config Load
    print("\n[*] Verifying Pydantic Settings Configuration:")
    try:
        sys.path.insert(0, base_dir)
        from app.config import settings
        print(f"  [OK] App Name         : {settings.APP_NAME}")
        print(f"  [OK] App Environment  : {settings.APP_ENV}")
        print(f"  [OK] Database URL     : {settings.DATABASE_URL}")
        print(f"  [OK] Default CF Weight: {settings.DEFAULT_WEIGHT_CF}")
        print(f"  [OK] Content Weight   : {settings.DEFAULT_WEIGHT_CONTENT}")
        print(f"  [OK] Location Weight  : {settings.DEFAULT_WEIGHT_LOCATION}")
        print(f"  [OK] Quality Weight   : {settings.DEFAULT_WEIGHT_QUALITY}")
        print(f"  [OK] MMR Lambda       : {settings.DEFAULT_MMR_LAMBDA}")
        print(f"  [OK] Max Distance (km): {settings.DEFAULT_MAX_DISTANCE_KM}")
    except Exception as e:
        print(f"  [FAIL] Settings loading failed: {e}")
        all_packages_ok = False
        
    # 5. Check FastAPI App Instantiation
    print("\n[*] Verifying FastAPI App Instantiation:")
    try:
        from app.main import app
        print(f"  [OK] FastAPI Application '{app.title}' instantiated successfully.")
    except Exception as e:
        print(f"  [FAIL] FastAPI app instantiation failed: {e}")
        all_packages_ok = False
        
    print("\n" + "=" * 75)
    if all_packages_ok and not missing_dirs:
        print("RESULT: ALL PHASE 1 ENVIRONMENT CHECKS PASSED SUCCESSFULLY!")
    else:
        print("RESULT: SOME CHECKS FAILED. PLEASE REVIEW LOGS ABOVE.")
    print("=" * 75)

if __name__ == "__main__":
    run_environment_check()
