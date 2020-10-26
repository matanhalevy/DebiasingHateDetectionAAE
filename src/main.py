from src.data_cleaning import GabPreparer, StormfrontPreparer, WaseemTwitterPreparer, DavidsonTwitterPreparer, \
    FountaTwitterPreparer, GolbeckTwitterPreparer


def main():
    # todo get this to run from shell with arg parsing
    # gab_perparer = GabPreparer(path_to_raw='../data/gab/GabHateCorpus_annotations.tsv', verbose=True)
    # gab_perparer.load_data()
    # gab_perparer.prepare_data()
    #
    # stormfront_preparer = StormfrontPreparer(path_to_raw='../data/stormfront/annotations_metadata.csv', verbose=True)
    # stormfront_preparer.load_data()
    # stormfront_preparer.prepare_data()

    waseem_preparer = WaseemTwitterPreparer(overwrite=True, verbose=True, path_to_raw=['../data/twitter_datasets/waseem/NAACL_SRW_2016.csv', '../data/twitter_datasets/waseem/NLP+CSS_2016.csv'])
    waseem_preparer.load_data()
    waseem_preparer.prepare_data()

    davidson_preparer = DavidsonTwitterPreparer(overwrite=True, verbose=True, path_to_raw='../data/twitter_datasets/davidson/davidson_labelled.csv')
    davidson_preparer.load_data()
    davidson_preparer.prepare_data()

    founta_preparer = FountaTwitterPreparer(overwrite=True, verbose=True, path_to_raw='../data/twitter_datasets/founta/hatespeechtwitter.csv')
    founta_preparer.load_data()
    founta_preparer.prepare_data()

    golbeck_preparer = GolbeckTwitterPreparer(overwrite=True, verbose=True, path_to_raw='../data/twitter_datasets/golbeck/onlineHarassmentDataset.csv')
    golbeck_preparer.load_data()
    golbeck_preparer.prepare_data()

if __name__ == '__main__':
    main()
