from __future__ import annotations

from fastapi import (
    HTTPException,
    Security,
    status,
)
from fastapi.security import APIKeyHeader
from lum.wml.rest import config, schema, env
from lum.clu.processors.document import Document as CluDocument, Sentence as CluSentence
from lum.clu.odin.mention import (
    Mention,
    TextBoundMention,
    CrossSentenceMention,
    EventMention,
    RelationMention,
)
import httpx
import typing

__all__ = ["WMLUtils"]


class WMLUtils:
    # see https://stackoverflow.com/a/74401249
    async def get_client():
        # create a new client for each request
        async with httpx.AsyncClient(
            timeout=config.WML_DEFAULT_TIMEOUT, follow_redirects=True
        ) as client:
            # yield the client to the endpoint function
            yield client
            # close the client when the request is done

    API_KEY_HEADER_NAME = "X-API-Key"
    api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME)

    @staticmethod
    def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
        """Validates API key"""
        if api_key_header in env.API_KEYS:
            return api_key_header
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

    @staticmethod
    def response_contains_mentions(jdata: dict[str, typing.Any]) -> bool:
        """Used to validate response from processors_proxy.extract_mentions_from_document"""
        if "documents" not in jdata:
            # TODO: consider additional logging
            print(f"No mentions found.")
            return False
        return True

    @staticmethod
    def system_from_env() -> None:
        raise NotImplementedError("WMLUtils.system_from_env() has not been implemented")

    @staticmethod
    def filter_extractions(
        extractions: list[schema.Extraction],
    ) -> list[schema.Extraction]:
        # FIXME implement filters for duplicate extractions
        print(f"Received {len(extractions)}")
        is_duration_extraction = lambda ex: (
            ex.variable == schema.ExtractionVariable.PROJECT_DURATION
            or ex.variable == schema.ExtractionVariable.ADDENDA_PROJECT_DURATION
        )
        measure_duration = lambda ex: (float("".join(filter(str.isdigit, ex.value))))
        max_duration = (-1, None)
        other = []
        for extraction in extractions:
            # Clean values
            extraction.value = WMLUtils.postprocess_evidence(extraction.value)
            # Clean Context\
            if extraction.eicontext != None:
                extraction.eicontext = WMLUtils.postprocess_evidence(
                    extraction.eicontext
                )
            # Filter Project Duration extractions
            if is_duration_extraction(extraction):
                n = measure_duration(extraction)
                if n > max_duration[0]:
                    max_duration = (n, extraction)
            else:
                other.append(extraction)
        longest: float = max_duration[-1]
        return other + [longest] if longest is not None else other

    @staticmethod
    def filter_mentions_with_metadata(
        mwms: list[schema.MentionWithMetadata],
    ) -> typing.Iterable[schema.MentionWithMetadata]:
        """Filters out undesired mentions. For example mentions flagged keep=false.
        Some filtering must take place before we produce extractions.
        """
        for mwm in mwms:
            if mwm.mention.keep == True:
                yield mwm

    @staticmethod
    def to_extractions(
        mwms: list[schema.MentionWithMetadata],
    ) -> list[schema.Extraction]:
        """Takes a list of mentions and passes each of them to Extraction.from_mention_with_metadata,
        to convert them to extractions.
        """
        extractions = []
        for mwm in WMLUtils.filter_mentions_with_metadata(mwms):
            for extraction in schema.Extraction.from_mention_with_metadata(mwm):
                extractions.append(extraction)
        return extractions

    @staticmethod
    def section_determiner(mns: list[Mention]) -> str:
        """section_determiner checks to see what the earliest match is for our
        section_grammar, presuming that a section name is likely to appear in the
        header or title of a doc. If there is no match in the first 200 characters,
        section-determiner returns "Unknown".
        """
        if not mns:
            return schema.SectionLabel.UNKNOWN
        else:
            # Looking at offsets
            offset_dict = {}
            for mn in mns:
                offset_dict[mn.char_start_offset] = mn.label
            key = min(offset_dict.keys())
            if key <= 200:
                return schema.SectionLabel.label_map(offset_dict[key])
            else:
                return schema.SectionLabel.UNKNOWN

    @staticmethod
    def from_mention_to_sentence(m: Mention) -> str:
        """Takes a mention and returns the sentence it was extracted from as a
        string.
        """
        startoffset = m.sentence_obj.start_offsets[0]
        endoffset = m.sentence_obj.end_offsets[-1]
        return m.document.text[startoffset:endoffset]

    @staticmethod
    def postprocess_evidence(context: str) -> str:
        """Filters problematic characters (newlines for now) out of context field"""
        return context.replace("\n", " ")
