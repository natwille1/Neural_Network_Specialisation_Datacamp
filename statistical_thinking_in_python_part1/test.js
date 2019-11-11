eventtype="et_acn_mywizard360_aiops_docker_demo_tier1"
// | rex field=host_docker "localhost:(?<host>[0-9]{4})" 
// | eval host="HOST"."-".host
// | stats perc90(value) as value by _time, host, counter
// | streamstats time_window=30s median(value) as median by counter, host
// | eval absDev = (abs('value'-median))
// | streamstats time_window=30s median(absDev) as medianAbsDev by counter, host
// | where time()-_time<30
// | eval lowerBound = (median-medianAbsDev*exact(2)), upperBound = (median+medianAbsDev*exact(2)), isOutlier = if(value < lowerBound OR value > upperBound, 1, 0), severity = absDev/medianAbsDev
// | lookup acn_mywizard360_aiops_configuration_output.csv host counter output type object
// | lookup data_topology_lookup counter output tier
// | eval threshold1=15, threshold2=15, threshold3=20, threshold4=20
// | eval alert_value = case( severity="TBC" , 2, severity <= threshold1, 1, severity >= threshold1 AND severity <= threshold2, 2, severity >= threshold2 AND severity <= threshold3, 3, severity >= threshold3 AND severity <= threshold4, 4, severity >= threshold4, 5)
// | eval host_counter='host'."-".'counter'
// | eventstats min(severity) AS minSev max(severity) AS maxSev
// | eval severity=round((severity-minSev)/(maxSev-minSev), 2), value = round(value, 1)
