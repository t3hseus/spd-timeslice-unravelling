import re
import collections

from src.metrics import *
from sklearn.cluster import KMeans
from src.transformations import ConstraintsNormalizer
from src.model import TrackEmbedder


def remove_spaces_from_keys(d):
    """Recursively remove spaces from dictionary keys."""
    if not isinstance(d, dict):
        return d
    return {k.replace(' ', ''): remove_spaces_from_keys(v) for k, v in d.items()}


def parse_gin(config_path: str) -> dict:
    """
    Parses a configuration file and returns a structured dictionary.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration structure.
    """

    def convert_value(value: str) -> any:
        """Converts string value to its appropriate type."""
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def extract_instance_names(s: str) -> list:
        """Extracts class or function names from a string."""
        return re.findall(r'@(\w+)\(\)', s)

    def get_instance_from_globals(name: str, **kwargs) -> any:
        """Retrieves an instance of a class or function from the global namespace."""
        instance = globals().get(name)
        if instance is None:
            raise ValueError(f"{name} not found in global namespace.")
        return instance(**kwargs) if kwargs else instance()

    cfg_struct: dict[str, dict[any]] = collections.defaultdict(dict)
    continue_symbol = "#"
    key_found_symbol = "="
    multi_line_value = "\\"
    last_first_key, last_second_key = None, None
    trash_line = ""

    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or continue_symbol in line:
                continue

            if key_found_symbol in line:
                if trash_line:
                    cfg_struct[last_first_key][last_second_key.strip()] = convert_value(
                        trash_line)
                    last_first_key, last_second_key = None, None
                    trash_line = ""

                first_key, second_key = line.split('=')[0].split('.')
                value = line.split('=')[1].strip().replace("'", "")
                if multi_line_value not in value:
                    cfg_struct[first_key][second_key] = convert_value(value)
                else:
                    last_first_key, last_second_key = first_key, second_key
                continue

            trash_line += line
    cfg_struct = remove_spaces_from_keys(cfg_struct)

    experiment = cfg_struct.get('experiment', {})
    keys_to_process = ['metrics', 'hits_normalizer', 'model']

    for key in keys_to_process:
        s = experiment.get(key)
        if s:
            names = extract_instance_names(s)
            if key == 'metrics':
                experiment[key] = [
                    get_instance_from_globals(name) for name in names]
            elif key == 'model' and 'TrackEmbedder' in cfg_struct:
                experiment[key] = get_instance_from_globals(
                    names[0], **cfg_struct['TrackEmbedder'])
            else:
                experiment[key] = get_instance_from_globals(names[0])

    cfg_struct['experiment'] = experiment
    return cfg_struct