import pandas as pd
from lum.wml.pdf.reader import PDFReader
import re
import os
from enum import StrEnum
import datetime


class SectionLabel(StrEnum):
    # Discussion of important sections: https://github.com/lum-ai/wml-issues/issues/13
    UNKNOWN = "UNKNOWN"
    TABLE_OF_CONTENTS = "TABLE OF CONTENTS"
    ADDENDA = "ADDENDA"
    BIDDING_AND_CONTRACT_DOCUMENTS = "BIDDING AND CONTRACT DOCUMENTS"
    GENERAL_REQUIRMENTS = "GENERAL REQUIREMENTS"
    EARTHWORK = "EARTHWORK"
    EXISTING_CONDITIONS = "EXISTING CONDITIONS"
    CONCRETE = "CONCRETE"
    METALS = "METALS"
    MASONRY = "MASONRY"
    WOOD_PLASTICS_COMPOSITES = "WOOD_PLASTICS_COMPOSITES"
    FINISHES = "FINISHES"
    EQUIPMENT = "EQUIPMENT"
    SPECIAL_CONSTRUCTION = "SPECIAL CONSTRUCTION"
    MECHANICAL = "MECHANICAL"
    ELECTRICAL = "ELECTRICAL"
    # INSTRUMENTATION_AND_CONTROLS may be equivalent to PROCESS_CONTROL
    INSTRUMENTATION_AND_CONTROLS = "INSTRUMENTATION AND CONTROLS"
    PROCESS_CONTROL = "PROCESS_CONTROL"
    SPECIALTIES = "SPECIALTIES"
    HVAC = "HVAC"
    EXTERIOR_IMPROVEMENTS = "EXTERIOR IMPROVEMENTS"
    UTILITIES = "UTILITIES"
    PROCESS_GAS_AND_LIQUID = "PROCESS GAS AND LIQUID"
    WATER_AND_WASTEWATER_EQUIPMENT = "WATER AND WASTEWATER EQUIPMENT"
    THERMAL_MOISTURE_PROTECTION = "THERMAL AND MOISTURE PROTECTION"
    OPENINGS = "OPENINGS"
    PROCESS_INTEGRATION = "PROCESS INTEGRATION"
    CONVEYING_SYSTEMS = "CONVEYING SYSTEMS"
    POLLUTION_AND_WASTEWATER_CONTROL_EQUIPMENT = (
        "POLLUTION AND WASTEWATER CONTROL EQUIPMENT"
    )
    ELECTRICAL_POWER_GENERATION = "ELECTRICAL POWER GENERATION"
    MATERIAL_PROCESSING_AND_HANDLING_EQUIPMENT = (
        "MATERIAL PROCESSING AND HANDLING EQUIPMENT"
    )
    PROCESS_HEATING_COOLING_DRYING_EQUIPMENT = (
        "PROCESS HEATING COOLING DRYING EQUIPMENT"
    )
    WATERWAY_AND_MARINE_CONSTRUCTION = "WATERWAY AND MARINE CONSTRUCTION"
    FURNISHING = "FURNISHING"
    PLUMBING = "PLUMBING"
    FIRE_SUPPRESSION = "FIRE SUPRESSION"
    INTEGRATED_AUTOMATION = "INTEGRATED AUTOMATION"
    COMMUNICATIONS = "COMMUNICATIONS"
    ELECTRONIC_SAFTEY_AND_SECURITY = "ELECTRONIC SAFTEY AND SECURITY"
    TRANSPORTATION = "TRANSPORTATION"
    MANUFACTURING_EQUIPMENT = "MANUFACTURING EQUIPMENT"

    def to_section_name(section_number: str) -> str:
        """Takes a 5 or six digit` Section Name and returns a Plain English Section label.
        5 digit Section Numbers were supplied by WML, 6 digit Section Numbers were found
        here : https://www.abc.org/Membership/MasterFormat-CSI-Codes-NAICS-Codes/CSI-Codes
        and confirmed to be correct by WML"""
        section_dict = {
            # 5 digit "old" section labels
            "00ddd": SectionLabel.BIDDING_AND_CONTRACT_DOCUMENTS,
            "01ddd": SectionLabel.GENERAL_REQUIRMENTS,
            "02ddd": SectionLabel.EARTHWORK,
            "03ddd": SectionLabel.CONCRETE,
            "04ddd": SectionLabel.MASONRY,
            "05ddd": SectionLabel.METALS,
            "06ddd": SectionLabel.WOOD_PLASTICS_COMPOSITES,
            "07ddd": SectionLabel.THERMAL_MOISTURE_PROTECTION,
            "08ddd": SectionLabel.OPENINGS,
            "09ddd": SectionLabel.FINISHES,
            "10ddd": SectionLabel.SPECIALTIES,
            "11ddd": SectionLabel.EQUIPMENT,
            "12ddd": SectionLabel.FURNISHING,
            "13ddd": SectionLabel.SPECIAL_CONSTRUCTION,
            "14ddd": SectionLabel.CONVEYING_SYSTEMS,
            "15ddd": SectionLabel.MECHANICAL,
            "16ddd": SectionLabel.ELECTRICAL,
            "17ddd": SectionLabel.INSTRUMENTATION_AND_CONTROLS,
            # 6 digit "new/CSI" section labels
            "00dddd": SectionLabel.BIDDING_AND_CONTRACT_DOCUMENTS,
            "01dddd": SectionLabel.GENERAL_REQUIRMENTS,
            "02dddd": SectionLabel.EXISTING_CONDITIONS,
            "03dddd": SectionLabel.CONCRETE,
            "04dddd": SectionLabel.MASONRY,
            "05dddd": SectionLabel.METALS,
            "06dddd": SectionLabel.WOOD_PLASTICS_COMPOSITES,
            "07dddd": SectionLabel.THERMAL_MOISTURE_PROTECTION,
            "08dddd": SectionLabel.OPENINGS,
            "09dddd": SectionLabel.FINISHES,
            "10dddd": SectionLabel.SPECIALTIES,
            "11dddd": SectionLabel.EQUIPMENT,
            "13dddd": SectionLabel.SPECIAL_CONSTRUCTION,
            "14dddd": SectionLabel.CONVEYING_SYSTEMS,
            "21dddd": SectionLabel.FIRE_SUPPRESSION,
            "22dddd": SectionLabel.PLUMBING,
            "23dddd": SectionLabel.HVAC,
            "25dddd": SectionLabel.INTEGRATED_AUTOMATION,
            "26dddd": SectionLabel.ELECTRICAL,
            "27dddd": SectionLabel.COMMUNICATIONS,
            "28dddd": SectionLabel.ELECTRONIC_SAFTEY_AND_SECURITY,
            "31dddd": SectionLabel.EARTHWORK,
            "32dddd": SectionLabel.EXTERIOR_IMPROVEMENTS,
            "33dddd": SectionLabel.UTILITIES,
            "34dddd": SectionLabel.TRANSPORTATION,
            "35dddd": SectionLabel.WATERWAY_AND_MARINE_CONSTRUCTION,
            "40dddd": SectionLabel.PROCESS_INTEGRATION,
            "41dddd": SectionLabel.MATERIAL_PROCESSING_AND_HANDLING_EQUIPMENT,
            "42dddd": SectionLabel.PROCESS_HEATING_COOLING_DRYING_EQUIPMENT,
            "43dddd": SectionLabel.PROCESS_GAS_AND_LIQUID,
            "44dddd": SectionLabel.POLLUTION_AND_WASTEWATER_CONTROL_EQUIPMENT,
            "45dddd": SectionLabel.MANUFACTURING_EQUIPMENT,
            "46dddd": SectionLabel.WATER_AND_WASTEWATER_EQUIPMENT,
            "48dddd": SectionLabel.ELECTRICAL_POWER_GENERATION,
        }
        if section_number in section_dict:
            return section_dict[section_number]
        else:
            return section_number


def collapse_section_number(section_number):
    if len(section_number) not in [5, 6]:
        return "Input  string must be 5 or 6 digits long"
    # Retain the first two digits and replace the rest with "d"
    modified_section_number = section_number[:2] + "d" * (len(section_number) - 2)
    return modified_section_number


def get_section_label(text: str) -> str:
    """Takes the text from a pdf page and returns a section label

    TODO:
        Improve Addenda functioning. Maybe at filename level? /ADD.*/ ?
    """
    # default section name to UNKNOWN
    section_name = SectionLabel.UNKNOWN
    # No examples of Section Numbers above 49
    add = "addend.*\s"
    digits = r"\b[0-4]\d\s?\d\d\s?\d\d?(?=\s|-)\b"
    toc = "\sTOC|Table\sof\sContents"

    fifth = len(text) // 5
    # check for label in header and footer
    header = re.findall(digits, text[0:fifth])
    footer = re.findall(digits, text[-fifth:-1])

    # check for short pages
    if len(text) < 1000:
        footer = re.findall(digits, text)

    # check for TOC in header/footer
    table_of_contents = re.findall(toc, text[0:fifth], flags=re.IGNORECASE)
    table_of_contents.extend(re.findall(toc, text[-fifth:-1], flags=re.IGNORECASE))

    # check for addenda in header/footer
    addendum = re.findall(add, text[0:fifth], flags=re.IGNORECASE)
    addendum.extend(re.findall(add, text[-fifth:-1], flags=re.IGNORECASE))

    # Sections ranked by order of importance
    if addendum:
        section_name = SectionLabel.ADDENDA
    elif table_of_contents:
        section_name = SectionLabel.TABLE_OF_CONTENTS
    # numerical section labels are usually in the footer
    elif footer:
        # get section number in "00dddd" or "00ddd" format
        section_number = collapse_section_number(footer[0].replace(" ", ""))
        # get section name in plain english
        section_name = SectionLabel.to_section_name(section_number)
        # replace section label in
    #  text = re.sub(footer[0], section_number, text)

    elif header:
        section_number = collapse_section_number(header[0].replace(" ", ""))
        # get section name in plain english
        section_name = SectionLabel.to_section_name(section_number)
        # replace section label in
    #  text = re.sub(header[0], section_number, text)
    return section_name, text


def confirm_continuous_section(
    df: pd.DataFrame, labels_changed: int, section_label: str, i: int
) -> pd.DataFrame:
    """Checks the last three values for df['SECTION_LABEL'], if the first and third
    labels are continuous, but the middle is not, resets the inner label to match the
    outer labels
    """
    last = df.loc[len(df) - 1]["SECTION_LABEL"]
    twoback = df.loc[len(df) - 2]["SECTION_LABEL"]
    # print(f"CURRENT: {section_label} \t LAST: {last}\t TWO AGO: {twoback} ")
    do_not_change_from = ["TABLE_OF_CONTENTS"]
    do_not_change_to = ["UNKNOWN"]
    # check for special cases
    if last in do_not_change_from or section_label in do_not_change_to:
        return df, labels_changed
    # make sections continuous
    if section_label == twoback:
        to_change = df.loc[len(df) - 1, "SECTION_LABEL"]
        print(f"Page {i} changed from {to_change} to {section_label}")
        df.loc[len(df) - 1, "SECTION_LABEL"] = section_label
        labels_changed += 1
    return df, labels_changed


def build_dataset(filepaths: list[str]) -> pd.DataFrame:
    """Takes a list of .pdf filepaths and produces a pandas DataFrame with the columns
    SECTION_LABEL, TEXT, SOURCE_DOCUMENT, with text being the text of a single page.
    If the end or beginning of the page contains a number in the format XXXXXX(-XX),
    then that listed as the section type.
    """
    df = pd.DataFrame(
        columns=["SOURCE_DOCUMENT", "PAGE_NO", "SECTION_LABEL", "TEXT", "URL"]
    )
    for path in sorted(filepaths):
        filename = os.path.basename(path)
        pdf = PDFReader.load(path)
        print(f"\n{filename}")
        labels_changed = 0
        for i, chunk in enumerate(PDFReader.chunk_pdf(pdf)):
            text = PDFReader.to_text(chunk, postprocess=True).replace("\n", " ")
            section_label, text = get_section_label(text)
            url = f"https://artifacts.wml.lum.ai/test-data/pdfs/{filename}#page={i+1}"

            # check if last page is same section as current
            if (i > 4) and (section_label != df.loc[len(df) - 1]["SECTION_LABEL"]):
                df, labels_changed = confirm_continuous_section(
                    df, labels_changed, section_label, i
                )
            new_row = [filename, (i + 1), section_label, text, url]
            df.loc[len(df)] = new_row
    print(f"TOTAL LABELS CHANGED: {labels_changed}")
    print("Number of Classes: {len(df['SECTION_LABEL'].unique())}")
    return df


source_dir = "/home/m/WML/tranq-adds"
alternate_path = "/home/m/WML/specs/bowman-specs.pdf"

if __name__ == "__main__":
    filepaths = []
    for file in os.listdir(source_dir):
        path = os.path.join(source_dir, file)
        filepaths.append(path)
    print()
    print(f"\nFILES = {sorted(filepaths)}\n")
    print()
    df = build_dataset(filepaths=filepaths)
    # df = build_dataset([alternate_path])
    # now = datetime.datetime.now().strftime("%m-%d_%H-%M")
    # output_file = (
    #     f"/home/m/git/lum/wml/python/lum/wml/data/section_training_data_{now}.csv"
    # )
    # df.to_csv(output_file)
