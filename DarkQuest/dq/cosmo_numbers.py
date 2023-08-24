

def convert_run_to_cosmo_number(run: int):
    if run > 100:
        return 0
    return (run+39)%100+1
