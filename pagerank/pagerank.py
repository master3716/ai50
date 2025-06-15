import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus:dict, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    num_pages = len(corpus)
    linked_pages = corpus[page]
    num_linked = len(linked_pages)

    dict_ranks = {}
    for p in corpus:
        dict_ranks[p] = (1 - damping_factor) / num_pages
    if num_linked == 0:
        for p in corpus:
            dict_ranks[p] += damping_factor / num_pages
    else:
        for p in linked_pages:
            dict_ranks[p] += damping_factor / num_linked
    return dict_ranks





def sample_pagerank(corpus:dict, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    dict_ranks = {}
    for p in corpus.keys():
        dict_ranks[p] = 0

    pages = list(corpus.keys())
    current_page = random.choice(pages)
    for i in range(n):
        dict_ranks[current_page] += 1

        sample = transition_model(corpus=corpus, page=current_page, damping_factor=damping_factor)

        current_page = pick_prob(sample)
    
    for key in dict_ranks.keys():
        dict_ranks[key] /= n


    return dict_ranks

def pick_prob(d: dict):
    r = random.random()
    cumulative = 0.0
    for key, prob in d.items():
        cumulative += prob
        if r <= cumulative:
            return key

def iterate_pagerank(corpus:dict, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    num_pages = len(corpus)
    pages = list(corpus.keys())
    

    dict_ranks = {page: 1 / num_pages for page in pages}
    
    converged = False
    while not converged:
        new_ranks = {}
        for page in pages:
           
            new_rank = (1 - damping_factor) / num_pages

            
            for possible_page in pages:
                if page in corpus[possible_page]:
                    num_links = len(corpus[possible_page])
                    if num_links == 0:
                        num_links = num_pages  
                    new_rank += damping_factor * dict_ranks[possible_page] / num_links
                elif len(corpus[possible_page]) == 0:
                    new_rank += damping_factor * dict_ranks[possible_page] / num_pages

            new_ranks[page] = new_rank

        converged = all(abs(new_ranks[p] - dict_ranks[p]) < 0.001 for p in pages)
        dict_ranks = new_ranks

    return dict_ranks

 




if __name__ == "__main__":
    main()
