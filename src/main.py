from src.data_cleaning.prepare_data import GabPreparer, StormfrontPreparer


def main():
    # todo get this to run from shell with arg parsing
    gab_perparer = GabPreparer(path_to_raw='../data/gab/GabHateCorpus_annotations.tsv', verbose=True)
    gab_perparer.load_data()
    gab_perparer.prepare_data()

    stormfront_preparer = StormfrontPreparer(path_to_raw='../data/stormfront/annotations_metadata.csv', verbose=True)
    stormfront_preparer.load_data()
    stormfront_preparer.prepare_data()


if __name__ == '__main__':
    main()
