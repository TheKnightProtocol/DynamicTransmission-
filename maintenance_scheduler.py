def maintenance_action(probability):

    if probability>0.90:

        return "Immediate Shutdown"

    elif probability>0.70:

        return "Schedule Maintenance"

    elif probability>0.40:

        return "Increase Monitoring"

    return "Normal Operation"
