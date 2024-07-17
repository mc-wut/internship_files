# -*- coding: utf-8 -*-
"""
Response models for API
"""
from __future__ import annotations
from dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict
from lum.wml.ie.odin import load_variable_grammar, load_section_grammar
from lum.clu.processors.document import Document as CluDocument
from lum.clu.odin.serialization import OdinJsonSerializer
from lum.clu.odin.mention import (
    Mention,
    TextBoundMention,
    CrossSentenceMention,
    EventMention,
    RelationMention,
)
from enum import StrEnum
import typing

__all__ = [
    "HealthStatus",
    "OdinPayload",
    "OdinMention",
    "Extraction",
    "ExtractionVariable",
    "SectionLabel",
    "MentionContext",
]


class SectionLabel(StrEnum):
    """SectionLabel is storage for section names, particularly important for Addenda."""

    UNKNOWN = "Unknown"
    ADDENDA = "Addenda"
    QUALIFICATIONS = "Qualifications"

    @staticmethod
    def label_map(arg_label: typing.Optional[str]) -> SectionLabel:
        """section_label_map maps a string to one of the values stored in SectionLabel"""
        for lbl in SectionLabel:
            if arg_label == lbl:
                return lbl
        # no match? return UNK
        return SectionLabel.UNKNOWN


class MentionContext(BaseModel):
    """MentionContext creates an object for LabelManager to use as a key to
    different variable labels based on Parent Label, Arg Label, and Section Name
    """

    PARENT_LABEL: typing.Optional[str] = Field(
        None, description="the label of parent mention"
    )
    ARG_LABEL: typing.Optional[str] = Field(
        None, description="the label of the target argument"
    )
    SECTION_LABEL: SectionLabel = Field(
        SectionLabel.UNKNOWN,
        description="Which section type the mention comes from",
    )
    # this will ensure we retrieve the string value of `SECTION_LABEL`
    model_config = ConfigDict(use_enum_values=True, validate_default=True)


class ExtractionVariable(StrEnum):
    """LabelManager is storage for variable names. It allows us to use the same reference
    for "Engineer of Record" everytime, so that typos and changes are not an issue.
    If anything needs to change about these variables we can change it here, rather
    than all over the repository.
    """

    ENGINEER_OF_RECORD = "Engineer of Record"
    BID_DATE = "Bid Date"
    BID_TIME = "Bid Time"
    PROJECT_DURATION = "Project Duration"
    ENGINEERS_ESTIMATE = "Engineer's Estimate"
    PROJECT_LOCATION = "Project Location"
    PROJECT_NAME = "Project Name"
    PROJECT_OWNER = "Project Owner"
    LIQUIDATED_DAMAGES = "Liquidated Damages"
    QUALIFICATIONS = "Qualifications"

    ADDENDA_PROJECT_DURATION = "Addenda Project Duration"
    ADDENDA_ENGINEERS_ESTIMATE = "Addenda Engineer's Estimate"
    ADDENDA_LIQUIDATED_DAMAGES = "Addenda Liquidated Damages"
    ADDENDA_BID_DATE = "Addenda Bid Date"
    ADDENDA_BID_TIME = "Addenda Bid Time"

    @staticmethod
    def label_map(mc: MentionContext) -> str:
        """label_map takes the labels extracted from an Event Mention in the form
        of a MentionContext and maps them to a standard variable name.
        In the case of a Text Bound Mention, some cases will be included in label_dict,
        if the case is not included, it will return the plain label.
        """
        label_dict = {
            ("BidDate", "Date", SectionLabel.UNKNOWN): ExtractionVariable.BID_DATE,
            (
                "BidDate",
                "ProperNoun",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.BID_DATE,
            ("BidTime", "Time", SectionLabel.UNKNOWN): ExtractionVariable.BID_TIME,
            (
                "Duration",
                "Date",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_DURATION,
            (
                "Duration",
                "CalendarDays",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_DURATION,
            (
                "Duration",
                "Number",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_DURATION,
            (
                "EngineersEstimate",
                "Dollars",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.ENGINEERS_ESTIMATE,
            (
                "EngineersEstimate",
                "UponRequest",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.ENGINEERS_ESTIMATE,
            (
                "ProjectLocation",
                "CompoundLocation",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_LOCATION,
            (
                "ProjectLocation",
                "Location",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_LOCATION,
            (
                "CompoundLocation",
                "Location",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_LOCATION,
            (
                "ProjectLocation",
                "CompoundLocation",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.PROJECT_LOCATION,
            (
                "ProjectLocation",
                "Location",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.PROJECT_LOCATION,
            (
                "CompoundLocation",
                "Location",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_LOCATION,
            (
                "ProjectNameEvent",
                "Project",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_NAME,
            (
                None,
                "ProjectName",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_NAME,
            (
                None,
                "EngineerOfRecord",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.ENGINEER_OF_RECORD,
            (
                "ProjectOwner",
                "Organization",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_OWNER,
            (
                None,
                "ProjectOwner",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.PROJECT_OWNER,
            (
                "ProjectOwner",
                "Organization",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.PROJECT_OWNER,
            (
                None,
                "ProjectOwner",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.PROJECT_OWNER,
            (
                None,
                "LiquidatedDamages",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.LIQUIDATED_DAMAGES,
            (
                "EngineersEstimate",
                "Dollars",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.ADDENDA_ENGINEERS_ESTIMATE,
            (
                "Duration",
                "Date",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.ADDENDA_PROJECT_DURATION,
            (
                "Duration",
                "CalendarDays",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.ADDENDA_PROJECT_DURATION,
            (
                "Duration",
                "Number",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.ADDENDA_PROJECT_DURATION,
            (
                None,
                "LiquidatedDamages",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.ADDENDA_LIQUIDATED_DAMAGES,
            (
                "BidMove",
                "Date",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.ADDENDA_BID_DATE,
            (
                "BidDate",
                "Date",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.ADDENDA_BID_DATE,
            (
                "BidMove",
                "Time",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.ADDENDA_BID_TIME,
            (
                "BidTime",
                "Time",
                SectionLabel.ADDENDA,
            ): ExtractionVariable.ADDENDA_BID_TIME,
            (
                None,
                "Integrator",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.QUALIFICATIONS,
            (
                None,
                "Integrator",
                SectionLabel.QUALIFICATIONS,
            ): ExtractionVariable.QUALIFICATIONS,
            (
                None,
                "Qualifications",
                SectionLabel.QUALIFICATIONS,
            ): ExtractionVariable.QUALIFICATIONS,
            (
                None,
                "PreQualification",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.QUALIFICATIONS,
            (
                None,
                "EngineerOfRecord",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.ENGINEER_OF_RECORD,
            (
                "EngineerOfRecord",
                "Org",
                SectionLabel.UNKNOWN,
            ): ExtractionVariable.ENGINEER_OF_RECORD,
        }

        if (mc.PARENT_LABEL, mc.ARG_LABEL, mc.SECTION_LABEL) in label_dict:
            return label_dict[(mc.PARENT_LABEL, mc.ARG_LABEL, mc.SECTION_LABEL)]
        elif mc.PARENT_LABEL == None:
            return mc.ARG_LABEL
        else:
            raise ValueError(
                f"[ {mc.PARENT_LABEL}, {mc.ARG_LABEL}, {mc.SECTION_LABEL} ] \
                    is not a valid combination . . . something's gone wrong."
            )


OdinMention = typing.Union[
    TextBoundMention, CrossSentenceMention, EventMention, RelationMention
]


class HealthStatus(BaseModel):
    wml: int = Field(description="HTTP status code for WML service", ge=100, le=599)
    processors: int = Field(
        description="HTTP status code for processors REST API proxy", ge=100, le=599
    )
    odinson: int = Field(
        description="HTTP status code for odinson REST API proxy", ge=100, le=599
    )


class OdinPayload(BaseModel):
    document: CluDocument
    rules: str = Field(
        default=load_variable_grammar(),
        description="The extraction grammar to apply to the provided doc",
    )

    @staticmethod
    def from_variable_grammar(doc: CluDocument) -> OdinPayload:
        variable_rules = load_variable_grammar()
        return OdinPayload(document=doc, rules=variable_rules)

    @staticmethod
    def from_section_grammar(doc: CluDocument) -> OdinPayload:
        section_rules = load_section_grammar()
        return OdinPayload(document=doc, rules=section_rules)


class MentionWithMetadata(BaseModel):
    mention: OdinMention = Field(description="An Odin Mention.")
    document_name: typing.Optional[str] = Field(
        default=None,
        description="Name of PDF document from which this extraction was produced.",
    )
    section: typing.Optional[str] = Field(
        default=None,
        description="Name of PDF document section from which this extraction was produced.",
    )
    page: typing.Optional[int] = Field(
        default=None, description="PDF page number where this extraction was found."
    )


class Extraction(BaseModel):
    variable: ExtractionVariable = Field(
        description="The variable for this extraction."
    )
    value: typing.Union[str, int] = Field(description="Value of the variable.")
    document_name: typing.Optional[str] = Field(
        default=None,
        description="Name of PDF document from which this extraction was produced.",
    )
    section: SectionLabel = Field(
        default=None,
        description="Name of PDF document section from which this extraction was produced.",
    )
    page: typing.Optional[int] = Field(
        default=None, description="PDF page number where this extraction was found."
    )
    eicontext: typing.Optional[str] = Field(
        default=None,
        description="Extraction in context, returns the sentence in which this extraction was found.",
    )
    provenance: typing.Optional[str] = Field(
        default=None,
        description="Name of the model (rule, LLM version, etc.) which produced the extraction.",
    )
    model_config = ConfigDict(use_enum_values=True, validate_default=True)

    @staticmethod
    def from_mention_with_metadata(
        mwm: MentionWithMetadata,
    ) -> typing.Iterator[Extraction]:
        """Takes mention_with_metadata and converts it to one or more extractions
        In certain cases a certain mention might result in multiple extractions,
        for instance an event mention with multiple arguments.
        """
        m = mwm.mention
        # FIXME: add EventMention etc.
        if not isinstance(m, TextBoundMention):
            for arg_name, args in m.arguments.items():
                # This is an argument we want to keep / convert to an extraction
                if not arg_name.startswith("dummy_"):
                    for arg in args:
                        print(m.label, arg.label)
                        # NOTE: This should be rewritten to be a recursive call to unpack nested mentions with args
                        ex = Extraction(
                            variable=ExtractionVariable.label_map(
                                MentionContext(
                                    PARENT_LABEL=m.label,
                                    ARG_LABEL=arg.label,
                                    SECTION_LABEL=mwm.section,
                                )
                            ),
                            value=arg.text,
                            document_name=mwm.document_name,
                            section=mwm.section,
                            provenance=m.found_by,
                            page=mwm.page,
                            # Use mention.sentence_obj's start and end offsets to get the full sentence
                            eicontext=m.document.text[
                                m.sentence_obj.start_offsets[
                                    0
                                ] : m.sentence_obj.end_offsets[-1]
                            ],
                        )
                        yield ex
        else:
            ex = Extraction(
                variable=ExtractionVariable.label_map(
                    MentionContext(ARG_LABEL=m.label)
                ),
                value=m.text,
                document_name=mwm.document_name,
                section=mwm.section,
                provenance=m.found_by,
                page=mwm.page,
                # Use mention.sentence_obj's start and end offsets to get the full sentence
                eicontext=m.document.text[
                    m.sentence_obj.start_offsets[0] : m.sentence_obj.end_offsets[-1]
                ],
            )
            yield ex
