import csv
import itertools
import math
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people: dict, one_gene:set, two_genes:set, have_trait:set):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    
    prob_dict = {}
    for p in people.keys():
        mother = people[p]["mother"]
        father = people[p]["father"]
        p_g = 1 if p in one_gene else (2 if p in two_genes else 0)
        p_chance = 0
        if(mother and father):
            mother_g = 1 if mother in one_gene else (2 if mother in two_genes else 0)
            father_g = 1 if father in one_gene else (2 if father in two_genes else 0)
        
            p_chance = prob_child_gene_count(p_g, mother_g, father_g) * PROBS["trait"][p_g][p in have_trait]
        else:
            p_chance = PROBS["gene"][p_g] * PROBS["trait"][p_g][p in have_trait]

        prob_dict[p] = p_chance

    joint_prob = 1.0
    for p in prob_dict:
        joint_prob *= prob_dict[p]
    return joint_prob

def get_pass_probability(g):
    if g == 2:
        return 1 - PROBS["mutation"]
    elif g == 1:
        return 0.5
    else:
        return PROBS["mutation"]

def prob_child_gene_count(child_genes, mother_g, father_g):
    mom_p = get_pass_probability(mother_g)
    dad_p = get_pass_probability(father_g)

    if child_genes == 2:
        return mom_p * dad_p
    elif child_genes == 1:
        return mom_p * (1 - dad_p) + (1 - mom_p) * dad_p
    else: 
        return (1 - mom_p) * (1 - dad_p)



def update(probabilities:dict, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for people in probabilities.keys():
        probabilities[people]["gene"][1 if people in one_gene else (2 if people in two_genes else 0)] += p
        probabilities[people]["trait"][people in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for people in probabilities.keys():
        s1 = sum(probabilities[people]["gene"].values())
        s2 = sum(probabilities[people]["trait"].values())
        
        alpha1 = 1 / s1
        alpha2 = 1 / s2

       
        probabilities[people]["gene"][0] *= alpha1
        probabilities[people]["gene"][1] *= alpha1
        probabilities[people]["gene"][2] *= alpha1

        probabilities[people]["trait"][True] *= alpha2
        probabilities[people]["trait"][False] *= alpha2
            

def check(family, one, two, trait):
    print(joint_probability(family, one, two, trait))

if __name__ == "__main__":
    main()
    #check(load_data("data/family0.csv"), set(["Harry"]), set(["James"]), set(["Harry", "James"]))
