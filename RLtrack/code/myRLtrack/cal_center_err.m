function distance = cal_center_err(a,b)
    tmp = (a(1)-b(1))^2 + (a(2)-b(2))^2;
    distance = sqrt(tmp);
end