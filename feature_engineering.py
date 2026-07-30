def create_features(df):

    df["Temperature_Difference"]=(

        df["Process temperature [K]"]

        -df["Air temperature [K]"]

    )

    df["Power"]=df["Torque [Nm]"]*df["Rotational speed [rpm]"]

    df["Wear_per_RPM"]=df["Tool wear [min]"]/df["Rotational speed [rpm]"]

    return df
