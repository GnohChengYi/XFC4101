# incorrect approach to evaluate the expected rmse for lerp


T = 10
missing_fraction = 0.9
m_r = int((T - 1) * missing_fraction)


# calculate the probability of (t1, t2) being the keyframes
# i.e. every frame between t1 and t2 is missing
# a specific t between t1 and t2 is given to be missing
def p(t1, t2):
    # print(t1, t2)
    d = t2 - t1
    num_to_remove = m_r - 1 # m_r - t
    num_maybe_remove = (T+1) - 3 # 0..T - {0, T} - t
    # if t1 != 0: num_unsure_missingness -= 1
    # if t2 != T: num_unsure_missingness -= 1
    
    # if num_to_remove <= 0 or num_unsure_missingness <= 0:
    #     return 0
    
    
    # t1
    result = 1 - num_to_remove / num_maybe_remove
    num_to_remove -= 1
    num_maybe_remove -= 1
    
    # t2
    result *= 1 - num_to_remove / num_maybe_remove
    num_to_remove -= 1
    num_maybe_remove -= 1
    
    for _ in range(1, d):
        # print(i, num_to_remove, num_unsure_missingness)
        if num_to_remove <= 0 or num_maybe_remove <= 0:
            break
        result *= num_to_remove / num_maybe_remove
        num_to_remove -= 1
        num_maybe_remove -= 1
    # print(f'p({t1},{t2})={result}')
    return result


# squared error when interpolating t from t1 and t2
def get_squared_error(t, t1, t2, f):
    left_value = f(t1)
    right_value = f(t2)
    interpolated_value = left_value + (right_value - left_value) * (t - t1) / (t2 - t1)
    return (interpolated_value - f(t)) ** 2


f = lambda t: 0 + 0 * t + 0.5 * 0.02 * t**2

sum_of_squares = 0
one = 0
for t in range(1, T):
    # one = 0
    for t1 in range(t):
        # print(f't1={t1}')
        for t2 in range(t+1, T+1):
            sum_of_squares += p(t1, t2) * get_squared_error(t, t1, t2, f)
            one += p(t1, t2)
    print(f'one:{one}')

rmse = (sum_of_squares / (T+1)) ** 0.5

print(rmse)

