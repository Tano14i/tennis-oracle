import unittest
import pandas as pd
from tennis_rank_history import carry_forward_prior_ranks

def row(day, w="a", l="b", wr=None, lr=None, tour="ATP"):
    return {"tourney_date": day, "tour": tour,
            "winner_id": w, "loser_id": l,
            "winner_rank": wr, "loser_rank": lr}

class RankHistoryTests(unittest.TestCase):
    def test_prior_day_rank_can_fill_missing(self):
        df=pd.DataFrame([
            row("2026-01-01", wr=15, lr=60),
            row("2026-01-03", wr=None, lr=None)
        ])
        out, stats=carry_forward_prior_ranks(df)
        self.assertEqual(out.loc[1,"winner_rank"],15)
        self.assertEqual(out.loc[1,"loser_rank"],60)
        self.assertEqual(stats["filled_from_prior_rank"],2)
        self.assertTrue(pd.isna(df.loc[1,"winner_rank"]))  # original unchanged

    def test_same_day_cannot_supply_rank(self):
        a=row("2026-01-01",w="a",l="b",wr=10,lr=50)
        b=row("2026-01-01",w="a",l="c",wr=None,lr=None)
        out,_=carry_forward_prior_ranks(pd.DataFrame([a,b]))
        self.assertTrue(pd.isna(out.loc[1,"winner_rank"]))
        out_reverse,_=carry_forward_prior_ranks(pd.DataFrame([b,a]))
        self.assertTrue(pd.isna(out_reverse.loc[0,"winner_rank"]))

    def test_rank_expires_after_70_days(self):
        out,_=carry_forward_prior_ranks(pd.DataFrame([
            row("2026-01-01",wr=11),row("2026-04-01")]))
        self.assertTrue(pd.isna(out.loc[1,"winner_rank"]))

    def test_imputation_does_not_refresh_staleness_clock(self):
        out,_=carry_forward_prior_ranks(pd.DataFrame([
            row("2026-01-01",wr=11),
            row("2026-02-01"),
            row("2026-04-01"),
        ]))
        self.assertEqual(out.loc[1,"winner_rank"],11)
        self.assertTrue(pd.isna(out.loc[2,"winner_rank"]))

    def test_tour_scoped_ids(self):
        out,_=carry_forward_prior_ranks(pd.DataFrame([
            row("2026-01-01",wr=11,tour="ATP"),
            row("2026-01-02",wr=None,tour="WTA"),
        ]))
        self.assertTrue(pd.isna(out.loc[1,"winner_rank"]))

    def test_disagreeing_same_day_rank_is_not_carried_forward(self):
        out,stats=carry_forward_prior_ranks(pd.DataFrame([
            row("2026-01-01",wr=12),
            row("2026-01-01",wr=33),
            row("2026-01-02"),
        ]))
        self.assertTrue(pd.isna(out.loc[2,"winner_rank"]))
        self.assertEqual(stats["source_rank_conflicts_same_day"],1)

if __name__=="__main__":
    unittest.main()
