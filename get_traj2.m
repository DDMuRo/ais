function trajectories = get_traj2(inputs, coeffs, w)
    
    S = w;

    t0 = coeffs(1);
    t1 = coeffs(2) * (inputs(:, 1)) * sin(S + coeffs(3));
    t2 = coeffs(4) * (inputs(:, 2)) * sin(S + coeffs(5));
    t3 = coeffs(6) * (inputs(:, 3)) * sin(S + coeffs(7));
    t4 = coeffs(8) * (inputs(:, 4)) * sin(S + coeffs(9));
    trajectories = t0 + t1 + t2 + t3 + t4;

end