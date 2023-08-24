import pandas as pd
from dq import PlanckCosmology, Cosmology
from dark_emulator.darkemu.gamma1 import gamma1_gp

cosmology = PlanckCosmology()
gamma1 = gamma1_gp()
gamma1.set_cosmology(cosmology)

def assign_bias(x):
    b1 = gamma1.get_bias(redshift=x.redshift, logdens=x.logn1)[0][0]
    b2 = gamma1.get_bias(redshift=x.redshift, logdens=x.logn2)[0][0]
    return b1*b2

def apply_bias(df, filename = 'train_xi_biased.csv'):
    df['b1b2'] = df.apply(assign_bias, axis=1)
    df['xi'] /= df['b1b2']
    df.to_csv(
            "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/" + filename,
            index=False,
    )

def get_growth_dict(df):
    redshifts = df['redshift'].unique()
    return {redshift: cosmology.Dgrowth_from_z(redshift) for redshift in redshifts}


def apply_growth(df, filename = 'train_xi_biased_growth_r2.csv'):
    growth_dict = get_growth_dict(df)
    df['growth'] = df['redshift'].map(growth_dict)
    df['xi'] /= df['growth']**2
    df.to_csv(
            "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/" + filename,
            index=False,
    )


test_df = pd.read_csv(
        "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_xi_biased_r2.csv"
)
test_df['xi'] *= test_df['r']**2/test_df['r']**0.5
test_df.to_csv(
        "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_xi_biased_rhalf.csv",
        index=False
)
#apply_bias(test_df, filename='test_xi_biased.csv')
#apply_growth(test_df, filename='test_xi_biased_growth_r2.csv')
val_df = pd.read_csv(
        "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_xi_biased_r2.csv"
)
val_df['xi'] *= val_df['r']**2/val_df['r']**0.5
val_df.to_csv(
        "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_xi_biased_rhalf.csv",
        index=False
)

#apply_bias(val_df, filename='val_xi_biased.csv')
#apply_growth(val_df, filename='val_xi_biased_growth_r2.csv')
train_df = pd.read_csv(
        "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_xi_biased_r2.csv"
)
#apply_bias(train_df, filename='train_xi_biased.csv')
#apply_growth(train_df, filename='train_xi_biased_growth_r2.csv')
train_df['xi'] *= train_df['r']**2/train_df['r']**0.5
train_df.to_csv(
        "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_xi_biased_rhalf.csv",
        index=False
)




'''
train_df.to_csv(
    "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/train_xi_biased_growth_r2.csv",
    index=False,
)
test_df.to_csv(
    "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/test_xi_biased_growth_r2.csv",
    index=False,
)
val_df.to_csv(
    "/cosma7/data/dp004/dc-cues1/DarkQuest/dataframes/val_xi_biased_growth_r2.csv",
    index=False,
)
'''

