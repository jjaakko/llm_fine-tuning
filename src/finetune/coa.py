"""Utilities to manipulate Chart of Accounts and extract excerpts from it."""

import json

import finetune.config as config
import finetune.const as const


def get_parents(
    coa_elements_all,
    code: str,
    include_account_name: bool = False,
    full_element=False,
) -> tuple[int, list]:
    """Gets parent elements or just names of the parents.

    Args:
        coa_elements_all (_type_): either elements for both balance sheet and income statement or just one of them.
        code (int): accointig code.
        include_account_name (bool, optional): _description_. Defaults to False.
        full_element (bool, optional): If true returns whole elements instead of elements' names . Defaults to False.
        depth (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # coa_elements_all can contain both balance sheet and income statement items.
    parents = []
    for i, coa_element in enumerate(coa_elements_all):
        for elem in coa_element:
            if str(elem.get("code", False)) == str(code):
                element = elem
                if include_account_name:
                    if full_element:
                        parents.append(element)
                    else:
                        parents.append(element["name"].capitalize())
                while True:
                    if element["parent"] is None:
                        return i, parents
                    element = coa_element[element["parent"]]
                    if full_element:
                        parents.append(element)
                    else:
                        parents.append(element["name"].capitalize())

    return i, parents


def get_childs_as_str(*args, **kwargs):
    coa_elements = get_childs(*args, **kwargs)
    coa_str = "\n".join(coa_elements)
    return coa_str


def get_childs(
    coa_elements, indexes, level_limit, forced_indexes=None, name_key: str = "name"
):
    """_summary_

    Args:
        coa_elements (_type_): _description_
        indexes (_type_): Starting position from the root: [0], or a specific parent[x]
        level_limit (_type_): _description_
        forced_indexes (_type_, optional): If this is specified, defines witch parents to include, parents that
        are not in the list will not be visited when descending the tree structure. Defaults to None.
        name_key (str): specifies the key used to pick the name of the account. Use "name" or "name_en" to get the original
        and translated names respectively.

    Returns:
        _type_: _description_
    """
    simple_coa = []

    def get_childs_(coa_elements, indexes, level):
        if level >= level_limit:
            return
        for index in indexes:
            if (
                forced_indexes
                and index not in forced_indexes
                and "code" not in coa_elements[index]
            ):
                # Indexes are forced, current index is not in the list, and it's not a leaf node.
                # We want to include leaf nodes without specifying them in forced_indexes.
                continue
            simple_coa.append(coa_elements[index])
            if "children" in coa_elements[index]:
                get_childs_(coa_elements, coa_elements[index]["children"], level + 1)

        return simple_coa

    simple_coa = get_childs_(coa_elements, indexes, level=0)

    return simple_coa


def get_coa_as_string_and_leaf_count(elements, name_key):
    lines = []
    leaf_count = 0
    elements_flattened = [elem for elems in elements for elem in elems]
    min_indent = elements_flattened[0]["indent"]
    for elem in elements_flattened:
        if "code" in elem:
            code = str(elem["code"]) + " "
            leaf_count += 1
        else:
            code = ""
        line = " " * (elem["indent"] - min_indent) + code + elem[name_key].capitalize()
        lines.append(line)
    return "\n".join(lines), leaf_count


def get_balance_and_income_statement_coa_elements(bodyid: int | str):
    datas = []
    for report_formula_name in const.REPORT_FORMULA_NAMES:
        filename = (
            config.data_path
            / "coa"
            / f"{bodyid}"
            / f"coa_elements_{report_formula_name}_{bodyid}.json"
        )

        with open(filename) as f:
            data = json.load(f)
            datas.append(data)
    return datas


def get_account_name_from_code(coa_elements_all, code, name_key: str = "name"):
    # coa_elements_all can contain both balance sheet and income statement items.
    account_name = "NOT_FOUND"
    for coa_element in coa_elements_all:
        for elem in coa_element:
            if elem.get("code", False) == code:
                account_name = elem[name_key].capitalize()
                break
    return account_name


def get_consolidated_elements_from_root(bodyid: str | int, level_limit: int = 2) -> str:
    """Get elements from root to leaf in the coa tree.

    Consolidates / combines both balance sheet and income statement elements.

    Example:
    To get the categories intended for the first, limiting prompt for llm, use:
    get_consolidated_elements_from_root(bodyid=123, level_limit=2)

    Args:
        bodyid (str): bodyid
        level_limit (int, optional): How many levels to include in the output.
        Use artbitraty large number to include all levels.

    Returns:
        str: Hierarchical string representation of the coa tree.
    """

    datas = get_balance_and_income_statement_coa_elements(bodyid)
    main_categories = get_consolidated_elements_from_root_by_data(
        datas, level_limit=level_limit
    )

    return "\n".join(main_categories[::-1])


def get_consolidated_elements_from_root_by_data(
    datas,
    level_limit: int = 2,
    indexes=[0],
    forced_indexes=None,
    name_key: str = "name",
):
    main_categories = []
    for data in datas:
        main_categories.append(
            get_childs(
                coa_elements=data,
                indexes=indexes,
                level_limit=level_limit,
                forced_indexes=forced_indexes,
                name_key=name_key,
            )
        )

    return main_categories[::-1]


def get_codes_from_coa(full_coa: list[list]):
    codes = []
    for coa in full_coa:
        for elem in coa:
            if "code" in elem:
                codes.append(elem["code"])
    return codes


def get_categories_and_leaf_count(instance, full_coa):
    i, parents = get_parents(
        full_coa,
        instance["account_code"],
        full_element=True,
    )
    forced_indexes = [parent["index"] for parent in parents]
    # Get all categories from the root, limited to specific indexes (parents of correct ledgeraccountcoude in this case)
    # all the way to the leaves.
    # full_coa[i] (0 or 1), points to balance_sheet or income_statement respectively
    categories = get_consolidated_elements_from_root_by_data(
        [full_coa[i]],
        indexes=[0],
        level_limit=100,
        forced_indexes=forced_indexes,
        name_key="name_en",
    )

    categories_str, number_of_targets = get_coa_as_string_and_leaf_count(
        categories, name_key="name_en"
    )
    return categories_str, number_of_targets


if __name__ == "__main__":
    code = 4021
    # get_parents(coa_elements_all, code: int, include_account_name: bool = False)
