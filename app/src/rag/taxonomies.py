from enum import Enum
from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import datetime

# ==========================================
# 1. TAXONOMY DEFINITIONS
# ==========================================

class PrimaryFocus(str, Enum):
    FARMED_ANIMALS = "farmed_animals"
    COMPANION_ANIMALS = "companion_animals"
    WILDLIFE = "wildlife"
    ANIMALS_IN_SCIENCE = "animals_in_science"
    INVERTEBRATES = "invertebrates"
    VEGAN_LIFESTYLE = "vegan_diet_and_lifestyle"
    SENTIENCE = "animal_sentience"
    ADVOCACY = "effective_advocacy"

class DocType(str, Enum):
    REPORT = "report"
    ARTICLE = "article"
    WEBSITE = "website"
    FORUM = "forum"
    ACADEMIC_PAPER = "academic_paper"

# ==========================================
# 2. METADATA SCHEMA
# ==========================================

class ChunkMetadata(BaseModel):
    source_name: str
    source_organization: str
    primary_focus: PrimaryFocus
    doc_type: DocType
    source_url: str = ""
    section: str = "" 
    
    source_hash: str 
    chunk_index: int
    chunk_id: str     
    
    # Keeping raw_date as a string/int to avoid nested dictionary errors
    raw_date: Union[str, int]
    publication_year: int = 0
    publication_date: str = "" 
    ingestion_date: str = Field(default_factory=lambda: datetime.date.today().isoformat())
    
    tags: List[str] = []

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """Normalizes all tags in the list to lowercase snake_case."""
        return [tag.lower().strip().replace(" ", "_") for tag in v]

    @model_validator(mode="after")
    def process_date_and_flatten(self) -> "ChunkMetadata":
        """
        Processes raw_date into flat fields and ensures raw_date 
        itself is a simple string, never a dictionary.
        """
        str_v = str(self.raw_date).strip()
        
        # Default fallback
        year = 1970
        date_str = "1970-01-01"

        if len(str_v) == 4 and str_v.isdigit():
            year = int(str_v)
            date_str = f"{str_v}-01-01"
        elif "-" in str_v:
            try:
                year = int(str_v.split("-")[0])
                date_str = str_v
            except (ValueError, IndexError):
                pass
        
        self.publication_year = year
        self.publication_date = date_str
        # Cast raw_date back to string to ensure the output is flat
        self.raw_date = str_v
        
        return self

    def to_db_json(self) -> Dict[str, Any]:
        """Returns a flat dictionary ready for Vector DB upsert."""
        return self.model_dump()

# ==========================================
# 3. USAGE
# ==========================================

if __name__ == "__main__":
    meta = ChunkMetadata(
        source_name="Global Animal Attitudes 2024",
        source_organization="Faunalytics",
        primary_focus=PrimaryFocus.ADVOCACY,
        doc_type=DocType.REPORT,
        source_hash="a1b2c3d4e5",
        chunk_index=4,
        chunk_id="a1b2c3d4e5_4",
        raw_date="2024-03-15",
        tags=["China", "Fish Welfare"]
    )
    
    # Output will be flat; no dictionaries in field values.
    print(meta.to_db_json())