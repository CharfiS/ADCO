import unittest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')  # chercher ADCO.py dans le même dossier

import ADCO


# ── Dataset de test connu ─────────────────────────────────────────────────
# Corrélations connues à la main :
#   r(A,B) = 1.0, r(A,C) = 1.0, r(B,C) = 1.0
#   r(A,E) ≈ 0.83, r(A,D) ≈ 0.21
# Groupes attendus avec seuil=0.8 : G1={A,B,C,E}, G2={D}

def make_test_df():
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6],
        "B": [2, 4, 6, 8, 10, 12],
        "C": [10, 20, 30, 40, 50, 60],
        "D": [5, 5, 6, 5, 6, 5],
        "E": [1, 3, 2, 4, 3, 5],
        "class": ["c1", "c2", "c1", "c2", "c1", "c2"]
    })


# ─────────────────────────────────────────────────────────────────────────────
class TestLoadDataset(unittest.TestCase):

    def test_load_sklearn_iris(self):
        """Iris via sklearn doit contenir 150 lignes et une colonne 'target'."""
        df = ADCO.load_dataset("iris")
        self.assertEqual(len(df), 150)
        self.assertIn("target", df.columns)

    def test_load_sklearn_wine(self):
        """Wine via sklearn doit contenir 178 lignes."""
        df = ADCO.load_dataset("wine")
        self.assertEqual(len(df), 178)

    def test_load_unknown_raises(self):
        """Un nom inconnu qui n'est pas un chemin CSV doit lever une erreur."""
        with self.assertRaises(Exception):
            ADCO.load_dataset("dataset_inexistant_xyz")

    def test_target_is_string(self):
        """La colonne target d'Iris doit contenir des noms de classes, pas des entiers."""
        df = ADCO.load_dataset("iris")
        # Accepte object ou StringDtype selon la version de pandas
        self.assertTrue(pd.api.types.is_string_dtype(df["target"]))
        self.assertIn("setosa", df["target"].values)


# ─────────────────────────────────────────────────────────────────────────────
class TestSplitDataset(unittest.TestCase):

    def setUp(self):
        self.df = ADCO.load_dataset("iris")

    def test_proportions(self):
        """Avec test_size=0.2, le test doit contenir ~20% des exemples."""
        df_train, df_test = ADCO.split_dataset(self.df, "target", test_size=0.2)
        self.assertAlmostEqual(len(df_test) / len(self.df), 0.2, delta=0.05)

    def test_no_overlap(self):
        """Les exemples train et test ne doivent pas se chevaucher."""
        df_train, df_test = ADCO.split_dataset(self.df, "target", test_size=0.2)
        # Comparer les lignes comme tuples de valeurs, pas les indices
        train_rows = set(df_train.apply(tuple, axis=1))
        test_rows  = set(df_test.apply(tuple, axis=1))
        self.assertEqual(len(train_rows & test_rows), 0)

    def test_total_size(self):
        """Train + test doit faire le total du dataset."""
        df_train, df_test = ADCO.split_dataset(self.df, "target", test_size=0.2)
        self.assertEqual(len(df_train) + len(df_test), len(self.df))

    def test_stratification(self):
        """Les proportions de classes doivent être respectées dans train et test."""
        df_train, df_test = ADCO.split_dataset(self.df, "target", test_size=0.2)
        for cls in self.df["target"].unique():
            ratio_train = (df_train["target"] == cls).sum() / len(df_train)
            ratio_test  = (df_test["target"]  == cls).sum() / len(df_test)
            self.assertAlmostEqual(ratio_train, ratio_test, delta=0.1)


# ─────────────────────────────────────────────────────────────────────────────
class TestAdcoGroups(unittest.TestCase):

    def setUp(self):
        self.df = make_test_df()

    def test_nombre_groupes(self):
        """Avec seuil=0.8, on doit obtenir exactement 2 groupes."""
        groups, _ = ADCO.adco_groups(self.df, target_col="class", threshold=0.8)
        self.assertEqual(len(groups), 2)

    def test_composition_groupes(self):
        """G1 doit contenir {A,B,C,E} et G2 doit contenir {D}."""
        groups, _ = ADCO.adco_groups(self.df, target_col="class", threshold=0.8)
        # Convertir en sets pour comparer sans ordre
        groups_sets = [set(g) for g in groups]
        self.assertIn({"A", "B", "C", "E"}, groups_sets)
        self.assertIn({"D"}, groups_sets)

    def test_target_exclue(self):
        """La colonne cible ne doit jamais apparaître dans les groupes."""
        groups, _ = ADCO.adco_groups(self.df, target_col="class", threshold=0.8)
        all_features = [f for g in groups for f in g]
        self.assertNotIn("class", all_features)

    def test_seuil_bas_un_groupe(self):
        """Avec seuil=0.0, tous les attributs doivent être dans un seul groupe."""
        groups, _ = ADCO.adco_groups(self.df, target_col="class", threshold=0.0)
        self.assertEqual(len(groups), 1)

    def test_seuil_haut_groupes_singletons(self):
        """Avec seuil=1.0, seuls A, B, C (corrélation=1.0) sont groupés ensemble."""
        groups, _ = ADCO.adco_groups(self.df, target_col="class", threshold=1.0)
        groups_sets = [set(g) for g in groups]
        self.assertIn({"A", "B", "C"}, groups_sets)
        self.assertIn({"D"}, groups_sets)
        self.assertIn({"E"}, groups_sets)

    def test_matrice_correlation(self):
        """r(A,B) doit être 1.0 et r(A,D) doit être < 0.3."""
        _, corr = ADCO.adco_groups(self.df, target_col="class", threshold=0.8)
        self.assertAlmostEqual(corr.loc["A", "B"], 1.0, places=4)
        self.assertLess(corr.loc["A", "D"], 0.3)


# ─────────────────────────────────────────────────────────────────────────────
class TestSimplifyConditions(unittest.TestCase):

    def test_redondance_superieure(self):
        """x <= 10 ET x <= 8 → x <= 8 (borne la plus stricte)."""
        conditions = [
            {"feature": "x", "op": "<=", "threshold": 10},
            {"feature": "x", "op": "<=", "threshold": 8},
        ]
        result = ADCO.simplify_conditions(conditions)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["threshold"], 8)

    def test_redondance_inferieure(self):
        """x > 2 ET x > 5 → x > 5 (borne la plus stricte)."""
        conditions = [
            {"feature": "x", "op": ">", "threshold": 2},
            {"feature": "x", "op": ">", "threshold": 5},
        ]
        result = ADCO.simplify_conditions(conditions)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["threshold"], 5)

    def test_contradiction(self):
        """x > 5 ET x <= 4 est une contradiction → None."""
        conditions = [
            {"feature": "x", "op": ">",  "threshold": 5},
            {"feature": "x", "op": "<=", "threshold": 4},
        ]
        result = ADCO.simplify_conditions(conditions)
        self.assertIsNone(result)

    def test_pas_de_redondance(self):
        """x > 2 ET x <= 8 : rien à simplifier, les deux conditions sont conservées."""
        conditions = [
            {"feature": "x", "op": ">",  "threshold": 2},
            {"feature": "x", "op": "<=", "threshold": 8},
        ]
        result = ADCO.simplify_conditions(conditions)
        self.assertEqual(len(result), 2)

    def test_plusieurs_attributs(self):
        """x > 1 ET x <= 5 ET y > 2 ET y <= 10 → 4 conditions conservées."""
        conditions = [
            {"feature": "x", "op": ">",  "threshold": 1},
            {"feature": "x", "op": "<=", "threshold": 5},
            {"feature": "y", "op": ">",  "threshold": 2},
            {"feature": "y", "op": "<=", "threshold": 10},
        ]
        result = ADCO.simplify_conditions(conditions)
        self.assertEqual(len(result), 4)

    def test_liste_vide(self):
        """Une liste vide doit retourner une liste vide."""
        result = ADCO.simplify_conditions([])
        self.assertEqual(result, [])


# ─────────────────────────────────────────────────────────────────────────────
class TestExtractRules(unittest.TestCase):

    def setUp(self):
        self.df = make_test_df()
        groups, _ = ADCO.adco_groups(self.df, target_col="class", threshold=0.8)
        self.trees = ADCO.train_trees(self.df, groups, target_col="class")

    def test_regles_non_vides(self):
        """Chaque arbre doit produire au moins une règle."""
        for tree_dict in self.trees:
            rules = ADCO.extract_rules(tree_dict)
            self.assertGreater(len(rules), 0)

    def test_structure_regle(self):
        """Chaque règle doit avoir les champs conditions, class, confidence, support."""
        for tree_dict in self.trees:
            for rule in ADCO.extract_rules(tree_dict):
                self.assertIn("conditions", rule)
                self.assertIn("class",      rule)
                self.assertIn("confidence", rule)
                self.assertIn("support",    rule)

    def test_confidence_entre_0_et_1(self):
        """La confiance doit toujours être entre 0 et 1."""
        for tree_dict in self.trees:
            for rule in ADCO.extract_rules(tree_dict):
                self.assertGreaterEqual(rule["confidence"], 0.0)
                self.assertLessEqual(rule["confidence"],    1.0)

    def test_support_positif(self):
        """Le support doit toujours être un entier strictement positif."""
        for tree_dict in self.trees:
            for rule in ADCO.extract_rules(tree_dict):
                self.assertIsInstance(rule["support"], int)
                self.assertGreater(rule["support"], 0)

    def test_classes_valides(self):
        """Les classes prédites doivent appartenir aux classes du dataset."""
        classes_valides = set(self.df["class"].unique())
        for tree_dict in self.trees:
            for rule in ADCO.extract_rules(tree_dict):
                self.assertIn(rule["class"], classes_valides)

    def test_pas_de_contradiction(self):
        """Aucune règle ne doit contenir de conditions contradictoires."""
        for tree_dict in self.trees:
            for rule in ADCO.extract_rules(tree_dict):
                features = [c["feature"] for c in rule["conditions"]]
                # Chaque feature ne doit apparaître qu'une fois en <= et une fois en >
                for feat in set(features):
                    lower = [c["threshold"] for c in rule["conditions"]
                             if c["feature"] == feat and c["op"] == ">"]
                    upper = [c["threshold"] for c in rule["conditions"]
                             if c["feature"] == feat and c["op"] == "<="]
                    if lower and upper:
                        self.assertLess(max(lower), min(upper))


# ─────────────────────────────────────────────────────────────────────────────
class TestBuildGlobalRules(unittest.TestCase):

    def setUp(self):
        self.df = make_test_df()
        groups, _ = ADCO.adco_groups(self.df, target_col="class", threshold=0.8)
        self.trees  = ADCO.train_trees(self.df, groups, target_col="class")
        self.GR     = ADCO.build_global_rules(self.trees)

    def test_champ_group_present(self):
        """Chaque règle de GR doit avoir un champ 'group'."""
        for rule in self.GR:
            self.assertIn("group", rule)

    def test_groupes_corrects(self):
        """Les groupes doivent être G1 et G2 uniquement."""
        groupes = {rule["group"] for rule in self.GR}
        self.assertEqual(groupes, {"G1", "G2"})

    def test_total_regles(self):
        """GR doit contenir autant de règles que la somme des règles par arbre."""
        total = sum(len(ADCO.extract_rules(t)) for t in self.trees)
        self.assertEqual(len(self.GR), total)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
