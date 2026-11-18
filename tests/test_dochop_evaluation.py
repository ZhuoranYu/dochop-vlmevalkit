import re

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vlmeval.dataset.dochop import DocHop
from vlmeval.smp.file import dump


@pytest.fixture
def evaluate():
    # Evaluation reads the saved predictions. Dataset initialization would
    # download images and is unnecessary for this path.
    return DocHop.__new__(DocHop).evaluate


def row(answer, prediction, **metadata):
    return dict(index=731, task='ranking', depth=2, chart_num=2, answer=answer, prediction=prediction) | metadata


def write_input(tmp_path, rows, suffix='csv'):
    path = tmp_path / f'fixture_DocHop.{suffix}'
    dump(pd.DataFrame(rows), str(path))
    return path


@pytest.mark.parametrize('suffix', ['csv', 'tsv', 'xlsx'])
def test_missing_reference_and_prediction_are_rejected(tmp_path, evaluate, suffix):
    path = write_input(tmp_path, [row(None, None)], suffix)
    with pytest.raises(ValueError, match=rf'{re.escape(str(path))}.*reference.*row 1.*731'):
        evaluate(str(path))
    assert list(tmp_path.iterdir()) == [path]


def test_missing_reference_with_valid_prediction_is_rejected(tmp_path, evaluate):
    path = write_input(tmp_path, [row(None, 'North team')])
    with pytest.raises(ValueError, match=r'reference.*row 1.*731'):
        evaluate(str(path))


@pytest.mark.parametrize('answer', ['', ' \t '], ids=['empty', 'whitespace'])
def test_blank_reference_is_rejected(tmp_path, evaluate, answer):
    path = write_input(tmp_path, [row(answer, 'North team')])
    with pytest.raises(ValueError, match=r'reference.*row 1.*731'):
        evaluate(str(path))


@pytest.mark.parametrize('missing', [('answer', ), ('prediction', ), ('answer', 'prediction')])
def test_required_columns_are_checked_before_scoring(tmp_path, evaluate, missing):
    record = row('North team', 'North team')
    for field in missing:
        del record[field]
    path = write_input(tmp_path, [record])
    with pytest.raises(ValueError, match=rf'{re.escape(str(path))}.*missing required.*{", ".join(missing)}'):
        evaluate(str(path))
    assert list(tmp_path.iterdir()) == [path]


def test_later_bad_reference_reports_position_and_sample_index(tmp_path, evaluate):
    path = write_input(tmp_path, [row('North team', 'North team'), row(None, 'South team', index=912)])
    with pytest.raises(ValueError, match=r'reference.*row 2.*912'):
        evaluate(str(path))
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('prediction', [None, pd.NA, '', ' \t '], ids=['none', 'pd-na', 'empty', 'whitespace'])
def test_blank_prediction_counts_in_denominator(tmp_path, evaluate, prediction):
    # Preserve raw null/empty/whitespace values for these distinct cases.
    path = write_input(tmp_path, [row('North team', prediction)], 'pkl')
    assert evaluate(str(path)).to_dict('records') == [{'Metric': 'Accuracy', 'Score': 0.0}]
    summary = pd.read_csv(tmp_path / 'fixture_DocHop_acc.csv')
    assert summary['Count'].tolist() == [1, 1, 1, 1]
    assert summary['Score'].tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize('answer,prediction', [('None', None), ('nan', float('nan'))], ids=['none', 'nan'])
def test_missing_prediction_cannot_match_literal_reference_text(tmp_path, evaluate, answer, prediction):
    # CSV/XLSX use pandas' NA-token parsing. Pickle preserves the literal
    # reference and the null value as distinct types for this test.
    path = write_input(tmp_path, [row(answer, prediction)], 'pkl')
    assert evaluate(str(path)).loc[0, 'Score'] == 0.0


def test_api_failure_sentinel_still_scores_incorrectly(tmp_path, evaluate):
    path = write_input(tmp_path, [row(0, 'Failed to obtain answer via API.')], 'xlsx')
    assert evaluate(str(path)).loc[0, 'Score'] == 0.0


def test_valid_answers_preserve_matching_groups_and_output_schema(tmp_path, evaluate):
    rows = [
        row(0, 0, task='counting'),
        row('North team', 'Working notes.\nThe answer is: **NORTH TEAM**.', chart_num=4),
        row('South team', 'Other team', depth=3, chart_num=4),
        row(100, 'The answer is: 100.9', task='numeric_reasoning', depth=3),
        row(100, '101.02', task='numeric_reasoning', depth=4, chart_num=6),
    ]
    path = write_input(tmp_path, rows, 'xlsx')

    assert_frame_equal(evaluate(str(path)), pd.DataFrame({'Metric': ['Accuracy'], 'Score': [60.0]}))
    details = pd.read_excel(tmp_path / 'fixture_DocHop_details.xlsx')
    assert details.columns.tolist() == [
        'index',
        'example_id',
        'task',
        'depth',
        'chart_num',
        'question',
        'answer',
        'prediction',
        'extracted',
        'correct',
    ]
    assert details['correct'].tolist() == [1, 1, 0, 1, 0]
    assert details.loc[1, 'extracted'] == 'NORTH TEAM'
    summary = pd.read_csv(tmp_path / 'fixture_DocHop_acc.csv')
    assert_frame_equal(
        summary,
        pd.DataFrame(
            [
                ('Overall', 'Total', 60.0, 5),
                ('Task', 'counting', 100.0, 1),
                ('Task', 'numeric_reasoning', 50.0, 2),
                ('Task', 'ranking', 50.0, 2),
                ('Depth', '2', 100.0, 2),
                ('Depth', '3', 50.0, 2),
                ('Depth', '4', 0.0, 1),
                ('Chart Num', '2', 100.0, 2),
                ('Chart Num', '4', 50.0, 2),
                ('Chart Num', '6', 0.0, 1),
            ],
            columns=['Dimension', 'Category', 'Score', 'Count'],
        ).astype({'Count': float}),
    )
