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
import json

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
        """Sorts out extractions with special filtering criteria (duration), and
        calls text cleaning and duplicate filtering methods
        """
        print(f"Received {len(extractions)}")
        durations = []
        addenda_durations = []
        filtered_extractions = []
        for extraction in extractions:
            # Clean values
            extraction.value = WMLUtils._postprocess_evidence(extraction.value)
            # Clean context
            if extraction.eicontext != None:
                extraction.eicontext = WMLUtils._postprocess_evidence(
                    extraction.eicontext
                )
            # Sort out Project Duration extractions
            if extraction.variable == schema.ExtractionVariable.PROJECT_DURATION:
                durations.append(extraction)
            elif (
                extraction.variable
                == schema.ExtractionVariable.ADDENDA_PROJECT_DURATION
            ):
                addenda_durations.append(extraction)
            else:
                filtered_extractions.append(extraction)
        # Filter for longest Project Duration and Addenda Duration
        if durations:
            filtered_extractions.append(WMLUtils._duration_filter(durations))
        if addenda_durations:
            filtered_extractions.append(WMLUtils._duration_filter(addenda_durations))
        return WMLUtils._filter_duplicate_extractions(filtered_extractions)

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
    def _postprocess_evidence(context: str) -> str:
        """Filters problematic characters (newlines for now) out of context field"""
        return context.replace("\n", " ")

    @staticmethod
    def _filter_duplicate_extractions(
        unfiltered_extractions: list[schema.Extraction],
    ) -> list[schema.Extraction]:
        """Removes duplicate extractions"""
        # Filter for duplicate extractions
        filtered_extractions = []
        variable_value_pairs = set()
        for ex in unfiltered_extractions:
            if (ex.variable, ex.value) not in variable_value_pairs:
                variable_value_pairs.add((ex.variable, ex.value))
                filtered_extractions.append(ex)
        return filtered_extractions

    @staticmethod
    def _duration_filter(
        extractions: list[schema.Extraction],
    ) -> list[schema.Extraction]:
        """Returns the longest duration extraction"""
        measure_duration = lambda ex: (float("".join(filter(str.isdigit, ex.value))))
        max_duration = (-1, None)
        for ex in extractions:
            n = measure_duration(ex)
            if n > max_duration[0]:
                max_duration = (n, ex)
        longest: schema.Extraction = max_duration[-1]
        return longest