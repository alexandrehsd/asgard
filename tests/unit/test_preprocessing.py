import numpy as np
from asgard.dataset.toolbox.preprocessing import preprocess_data

CSV_FILEPATH = "./tests/unit/test_data"

X = np.array(['COVID-19, school closures, and child poverty: a social crisis in the making',
              'A Social Vulnerability Index for Disaster Management',
              'Economic and social consequences of human mobility restrictions under COVID-19',
              'A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker)',
              'GIS-based spatial modeling of COVID-19 incidence rate in the continental United States',
              ""
              ])

y = np.random.randint(2, size=(6, 4))


def test_preprocess_data_truncation_lemma():
    X_processed, y_processed = preprocess_data(X, y, truncation="lemma")

    expected_X = np.array(['covid school closure child poverty social crisis make',
                           'social vulnerability index disaster management',
                           'economic social consequence human mobility restriction covid',
                           'global panel database pandemic policy oxford covid government response tracker',
                           'gisbase spatial model covid incidence rate continental united states'])

    assert y_processed.shape == (5, 4)
    assert len(X_processed) == 5
    assert (X_processed == expected_X).all()


def test_preprocess_data_truncation_stem():
    X_processed, y_processed = preprocess_data(X, y, truncation="stem")

    expected_X = np.array(['covid school closur child poverti social crisi make',
                           'social vulner index disast manag',
                           'econom social consequ human mobil restrict covid',
                           'global panel databas pandem polici oxford covid govern respons tracker',
                           'gisbas spatial model covid incid rate continent unit state'])

    assert y_processed.shape == (5, 4)
    assert len(X_processed) == 5
    assert (X_processed == expected_X).all()


def test_preprocess_data_truncation_none():
    X_processed, y_processed = preprocess_data(X, y, truncation=None)

    expected_X = np.array(['covid school closures child poverty social crisis making',
                           'social vulnerability index disaster management',
                           'economic social consequences human mobility restrictions covid',
                           'global panel database pandemic policies oxford covid government response tracker',
                           'gisbased spatial modeling covid incidence rate continental united states'])

    assert y_processed.shape == (5, 4)
    assert len(X_processed) == 5
    assert (X_processed == expected_X).all()
