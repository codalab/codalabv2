# pylint: disable=logging-fstring-interpolation, broad-except
"""autoWSL parent scoring"""
import os
from os.path import join
import sys
import logging
import argparse
from shutil import copyfile
import yaml

DEFAULT_NUM_DATASET = 18


def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


LOGGER = get_logger('INFO')
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SCORE = join(CURRENT_PATH, 'default_scores.txt')


def _arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./test_input',
                        help='where input results are stored')
    parser.add_argument('--output_dir', type=str, default='./test_output',
                        help='where to store aggregated outputs')
    args = parser.parse_args()
    LOGGER.debug(f"Parsed args are: {args}")
    LOGGER.debug("-" * 50)
    return args


def validate_full_res(args):
    """
    Check if we have DEFAULT_NUM_DATASET results in the args.input_dir.
    Replace by default values otherwise.
    """
    for i in range(DEFAULT_NUM_DATASET):
        # Check whether res_i/ exists
        check_path = join(args.input_dir, f"res_{i+2}")
        LOGGER.info(f"Checking {check_path}")
        if not os.path.exists(check_path):
            # Replace both learning curve and score by default:
            LOGGER.warning(
                f"{check_path} does not exist. Default values will be used.")
            # Create this folder and copy default values
            #  raise ValueError("failed in one dataset, stoped")
            os.makedirs(check_path)
            copyfile(DEFAULT_SCORE, join(check_path, "scores.txt"))
        else:
            # Replace either learning curve or score by default, depending...
            if not os.path.exists(join(check_path, "scores.txt")):
                logging.warning(
                    "Score file does not exist. Default values will be used.")
                copyfile(DEFAULT_SCORE, join(check_path, "scores.txt"))


def read_score(args):
    """
        Fetch scores from scores.txt
    """
    score_ls = []
    for i in range(DEFAULT_NUM_DATASET):
        score_dir = args.input_dir + "/res_"+str(i+2)
        score_file = join(score_dir, "scores.txt")
        try:
            with open(score_file, 'r') as ftmp:
                score_info = yaml.safe_load(ftmp)
            score_ls.append(float(score_info['score']))
        except Exception as ex:
            LOGGER.exception(f"Failed to load score in: {score_dir}")
            LOGGER.exception(ex)
    return score_ls


def write_score(score_ls, args):
    """
        Write scores to master phase scores.txt,
        as setj_score, where j = 1 to DEFAULT_NUM_DATASET
    """
    output_file = join(args.output_dir, 'scores.txt')
    try:
        with open(output_file, 'w') as ftmp:
            ftmp.write("score: \n")
            for i in range(DEFAULT_NUM_DATASET):
                #  score_name = f'set{i+1}_score'
                score_name = f's{i+1}'
                score = score_ls[i]
                ftmp.write(f"{score_name}: {score}\n")
    except Exception as ex:
        LOGGER.exception(f"Failed to write to {output_file}")
        LOGGER.exception(ex)


def write_curve(args):
    """
        Write learning curves concatenated
    """
    filename = 'detailed_results.html'
    detailed_results_path = join(args.output_dir, filename)
    html_head = '<html><body><pre>'
    html_end = '</pre></body></html>'
    try:
        with open(detailed_results_path, 'w') as html_file:
            html_file.write(html_head)
            html_file.write('<br>')
            html_file.write(html_end)
    except Exception as ex:
        LOGGER.exception(f"Failed to write to {detailed_results_path}")
        LOGGER.exception(ex)


def main():
    """main entry"""

    try:
        # Get input and output dir from input arguments
        args = _arg_parse()

        os.system(f"cp -R {join(args.input_dir, '*')} {args.output_dir}")

        if not os.path.exists(args.input_dir):
            raise RuntimeError("No input folder! Exit!")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # List the contents of the input directory
        # (should be a bunch of res_i/ subdirectories)
        input_ls = sorted(os.listdir(args.input_dir))
        LOGGER.debug(f"Input dir contains: {input_ls}")

        # Check if we have correct results in input_dir/res_i/
        # and copy default values otherwise
        validate_full_res(args)
        LOGGER.info("[+] Results validation done.")
        LOGGER.debug("-" * 50)
        LOGGER.debug("Start aggregation...")

        # Read all scores from input_dir/res_i/ subdirectories
        score_ls = read_score(args)
        LOGGER.info("[+] Score reading done.")
        LOGGER.info(f"Score list: {score_ls}")

        # Aggregate all scores and write to output
        write_score(score_ls, args)
        LOGGER.info("[+] Score writing done.")

        # Aggregate all learning curves and write to output
        write_curve(args)
        LOGGER.info("[+] Learning curve writing done.")
        LOGGER.info("[+] Parent scoring program finished!")

    except Exception as ex:
        LOGGER.exception(
            "Unexpected exception raised! Check parent scoring program!")
        LOGGER.exception(ex)


if __name__ == "__main__":
    main()
