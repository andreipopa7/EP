import unittest
from bayesian_network import BayesianNetwork

class TestBayesianNetwork(unittest.TestCase):

    def setUp(self):
        # Se incarca reteaua de test pentru fiecare test
        self.network = BayesianNetwork('test1_network.json')

    # Test pentru incarcarea retelei
    def test_load_network(self):
        self.assertIn('Gripa', self.network.network)
        self.assertIn('Abces', self.network.network)
        self.assertIn('Febra', self.network.network)

    # Test pentru obtinerea valorilor posibile
    def test_get_possible_values(self):
        values = self.network.get_possible_values('Gripa')
        self.assertListEqual(values, ['Da', 'Nu'])

    # Test pentru setarea evidentelor
    def test_set_evidence(self):
        evidence = {'Gripa': 'Da'}
        self.network.set_evidence(evidence)
        self.assertEqual(self.network.evidence, evidence)

    # Test pentru inferenta prin enumerare
    def test_enumeration_ask(self):
        self.network.set_evidence({'Gripa': 'Da'})
        result = self.network.enumeration_ask('Oboseala')
        self.assertAlmostEqual(result['Da'], 0.482, places=3)
        self.assertAlmostEqual(result['Nu'], 0.518, places=3)

    # Test pentru probabilitatea evidentelor
    def test_p_e_query(self):
        evidence = {'Febra': 'Da'}
        result = self.network.p_e_query(evidence)
        self.assertAlmostEqual(result, 0.1245, places=4)

    # Test pentru identificarea nodurilor irelevante
    def test_find_irrelevant_nodes(self):
        evidence = {'Febra': 'Nu'}
        irrelevant = self.network.find_irrelevant_nodes('Oboseala', evidence)
        self.assertIn('Gripa', irrelevant)
        self.assertIn('Abces', irrelevant)
        self.assertIn('Anorexie', irrelevant)

if __name__ == '__main__':
    unittest.main()