from numpy import ravel
import arch

class EWMA():
    def __init__(self, alpha=None, span=None):
        assert (alpha == None) + (span == None) == 1, "specify only exactly one parameter"
        
        if span == None:
            self.alpha = alpha
            self.span = (2 / alpha) - 1
        else:
            self.span = span
            self.alpha = 2 / (span + 1)
            
    def fit(self, time_series):
        result = time_series.ewm(span=self.span, adjust=False).mean()
        return result
    
    def forecast(self, time_series, span=20):
        forecast_value = self.fit(time_series=time_series)[-1]
        return forecast_value
    
class GARCH():
    pass
        