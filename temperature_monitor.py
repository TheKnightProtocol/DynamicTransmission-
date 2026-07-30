class TemperatureMonitor:

    SAFE_TEMP = 330

    WARNING_TEMP = 340

    CRITICAL_TEMP = 350


    @staticmethod
    def status(temp):

        if temp < TemperatureMonitor.SAFE_TEMP:
            return "Normal"

        elif temp < TemperatureMonitor.WARNING_TEMP:
            return "Warning"

        elif temp < TemperatureMonitor.CRITICAL_TEMP:
            return "High"

        return "Critical"
