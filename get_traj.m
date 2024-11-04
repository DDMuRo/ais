function trajectories = get_traj(inputs, coeffs, theta, f)
    
    S = 2*pi*theta*f;

    t0 = coeffs(1) + 20;
    t1 = coeffs(2) * (inputs(:, 1) ./ 10) * sin(S - coeffs(6));
    t2 = coeffs(3) * (inputs(:, 2) ./ 10) * sin(S - coeffs(7));
    t3 = coeffs(4) * (inputs(:, 3) ./ 10) * sin(S - coeffs(8));
    t4 = coeffs(5) * (inputs(:, 4) ./ 10) * sin(S - coeffs(9));
    trajectories = t0 + t1 + t2 + t3 + t4;

end