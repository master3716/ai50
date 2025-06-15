import random

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
    dict_ranks[page] = (1 - damping_factor) / num_pages
    for p in linked_pages:
        dict_ranks[p] = (damping_factor / num_linked) + ((1 - damping_factor) / num_pages)
    
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
    for i in range(n - 1):
        dict_ranks[current_page] += 1

        sample = transition_model(corpus=corpus, page=pages[pages.index(current_page)], damping_factor=damping_factor)

        current_page = pick_prob(sample)
    
    for key in dict_ranks.keys():
        dict_ranks[key] /= n


    return dict_ranks

def pick_prob(d:dict):
    while True:
        for key, value in d.items():
            rand = random.randint(1, 100) / 100
            if(rand <= value):
                return key

def iterate_pagerank(corpus: dict, damping_factor, tolerance=0.001):
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

        converged = all(abs(new_ranks[p] - dict_ranks[p]) < tolerance for p in pages)
        dict_ranks = new_ranks

    return dict_ranks



        
        
def main():
    c = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}
    d = transition_model(c, "1.html", 0.85)
    print(d)

    print("-------------------------")
    di = sample_pagerank(c, 0.85, 1000)
    print(di)
    s = 0
    for v in di.values():
        s += v
    print(s)

    print("--------------------------")

    dic = iterate_pagerank(c, 0.85)
    print(dic)
    s = 0
    for v in dic.values():
        s += v
    print(s)
main()