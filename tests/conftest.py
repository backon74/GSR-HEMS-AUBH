import pytest

from logic import runner
from logic.load_data import load_data


@pytest.fixture(scope='session')
def df():
    return load_data(verbose=False)


@pytest.fixture(scope='session')
def prepared(df):
    return runner.prepare(df)


@pytest.fixture(scope='session')
def fdf(prepared):
    return prepared[0]


@pytest.fixture(scope='session')
def sched(fdf):
    return runner.build_schedule(fdf)[0]


@pytest.fixture(scope='session')
def pipeline_outputs():
    import pipeline
    return pipeline.run_pipeline()
