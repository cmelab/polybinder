from . import smiles_utils

convert_smiles = smiles_utils.convert_smiles


def poly_smiles(monomer_string, length=2):
    """
    """
    brackets = count_brackets(monomer_string)
    monomer_list = list(monomer_string)
    # Check that the polymerization site was specified correctly
    if "*" not in monomer_list:
        raise ValueError("Identify the wanted polymerization site using *x*'")
    key_indices = [
        index for index, value in enumerate(monomer_list) if value == "*"
    ]
    if len(key_indices) != 2:  # Checks for only a single given poly site
        raise ValueError("Select only one polymerization site using *x*")
    # Check that the * are surrounding only a single atom
    if (key_indices[1] - key_indices[0] != 2):
        raise ValueError("Select only one polymerization site using *x*")

    # Set up the template string with {} and the correct # of brackets
    # Create poly site+brackets
    monomer_list[key_indices[1]] = "{}" + "{}".format(brackets)
    monomer_list.remove("*")
    # Deepsmiles string with {} at bonding site
    template = "".join(monomer_list)
    monomer_list.remove("{}" + "{}".format(brackets))
    # Deepsmiles monomer string without {} or *
    monomer = "".join(monomer_list)

    # Loop & format polymer
    polymer = "{}"
    for i in range(0, length):
        if i == length - 1:
            polymer = polymer.format(monomer)
            break
        polymer = polymer.format(template)

    polymer_smiles = convert_smiles(deep=polymer)
    return polymer_smiles


def count_brackets(deep_smiles_string):
    atom_count = 0
    bracket_count = 0
    for s in deep_smiles_string:
        if s.isalpha():
            atom_count += 1
        if s == ")":
            bracket_count += 1
    if bracket_count == 0:
        brackets = ")" * atom_count
    elif bracket_count != 0:
        brackets = ")" * (atom_count - bracket_count)
    return brackets
