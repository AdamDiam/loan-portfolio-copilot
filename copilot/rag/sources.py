"""Registry of the public regulatory documents indexed by the RAG module."""
from __future__ import annotations

from dataclasses import dataclass

from copilot.config import PROJECT_ROOT

REGULATION_DIR = PROJECT_ROOT / "docs" / "regulation"


@dataclass(frozen=True)
class Source:
    doc_id: str      # short id used in citations
    title: str
    filename: str
    url: str


SOURCES: list[Source] = [
    Source(
        doc_id="EBA/GL/2018/06",
        title="EBA Guidelines on management of non-performing and forborne exposures (2018)",
        filename="eba_gl_2018_06_npe_forborne.pdf",
        url=(
            "https://www.eba.europa.eu/sites/default/files/documents/10180/2425705/"
            "371ff4ba-d7db-4fa9-a3c7-231cb9c2a26a/"
            "Final%20Guidelines%20on%20management%20of%20non-performing%20and%20forborne%20exposures.pdf"
        ),
    ),
    Source(
        doc_id="EBA/GL/2016/07",
        title="EBA Guidelines on the application of the definition of default (2016)",
        filename="eba_gl_2016_07_definition_of_default.pdf",
        url=(
            "https://www.eba.europa.eu/documents/10180/1721448/"
            "Guidelines+on+default+definition+%28EBA-GL-2016-07%29_EN.pdf/052c260f-da9a-4c86-8f0a-09a1d8ae56e7"
        ),
    ),
    Source(
        doc_id="ECB NPL Guidance",
        title="ECB Guidance to banks on non-performing loans (2017)",
        filename="ecb_guidance_on_npl_2017.pdf",
        url="https://www.bankingsupervision.europa.eu/ecb/pub/pdf/guidance_on_npl.en.pdf",
    ),
]

SOURCES_BY_FILE = {s.filename: s for s in SOURCES}
