
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    print("Verifying imports...")
    from src.schemas.schemas import ProjectCreate, ProjectResponse, ConversationCreate, FileUpload
    print("✅ Schemas imported successfully")
    
    from src.models.models import Project
    print("✅ Models imported successfully")
    
    from src.api.v1.projects import router as projects_router
    print("✅ Projects router imported successfully")
    
    from src.api.v1.conversations import create_conversation
    print("✅ Conversations API imported successfully")
    
    import src.main
    print("✅ Main module imported successfully")
    
    print("All checks passed!")
    sys.exit(0)
except Exception as e:
    print(f"❌ Verification failed: {e}")
    sys.exit(1)
