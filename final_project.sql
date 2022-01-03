/******************************************************************************************************/
/*                        Huong Pham, Final Project, SQL code, Due 12/07/2021                         */
/******************************************************************************************************/
USE baseball;
SHOW tables;
set @@max_heap_table_size=104857600;

/*
 * CHECK OUT DATA 15890 games, 15890 home team, 15890 away team
 * FIX DATA Caughtstealing2b, 3b, home; Stolenbase2b, 3b, home; Strikeout_TP
 * CREATE ROLLING 100 DAYS, using the fixed team_batting_counts and fixed team_pitching_counts*
 * CREATE ROLLING 200 DAYS
 * CREATE ROLLING 365 DAYS
 * CREATE NEW FEATURES
 */


/**************************************************************************************
 *                                     FIX DATA
 *        Caughtstealing2b, 3b, home; Stolenbase2b, 3b, home; Strikeout_TP
 **************************************************************************************/

DROP TABLE IF EXISTS fix_stealing_a;
CREATE TABLE fix_stealing_a AS
SELECT * FROM
    (SELECT -- Away team
            g.game_id
            , g.away_team_id AS team_id
            , SUM(CASE WHEN des = 'Stolen Base 2B'       THEN 1 ELSE 0 END) AS stolenBase2B
            , SUM(CASE WHEN des = 'Stolen Base 3B'       THEN 1 ELSE 0 END) AS stolenBase3B
            , SUM(CASE WHEN des = 'Stolen Base Home'     THEN 1 ELSE 0 END) AS stolenBaseHome
            , SUM(CASE WHEN des = 'Caught Stealing 2B'   THEN 1 ELSE 0 END) AS caughtStealing2B
            , SUM(CASE WHEN des = 'Caught Stealing 3B'   THEN 1 ELSE 0 END) AS caughtStealing3B
            , SUM(CASE WHEN des = 'Caught Stealing Home' THEN 1 ELSE 0 END) AS caughtStealingHome
         FROM inning i
         JOIN game g ON g.game_id = i.game_id
         WHERE i.half = 0 AND i.entry = 'runner'
         GROUP BY g.game_id, g.away_team_id

     UNION
     SELECT -- Home team
            g.game_id
            , g.home_team_id AS team_id
            , SUM(CASE WHEN des = 'Stolen Base 2B'       THEN 1 ELSE 0 END) AS stolenBase2B
            , SUM(CASE WHEN des = 'Stolen Base 3B'       THEN 1 ELSE 0 END) AS stolenBase3B
            , SUM(CASE WHEN des = 'Stolen Base Home'     THEN 1 ELSE 0 END) AS stolenBaseHome
            , SUM(CASE WHEN des = 'Caught Stealing 2B'   THEN 1 ELSE 0 END) AS caughtStealing2B
            , SUM(CASE WHEN des = 'Caught Stealing 3B'   THEN 1 ELSE 0 END) AS caughtStealing3B
            , SUM(CASE WHEN des = 'Caught Stealing Home' THEN 1 ELSE 0 END) AS caughtStealingHome
         FROM inning i
         JOIN game g ON g.game_id = i.game_id
         WHERE i.half = 1 AND i.entry = 'runner'
         GROUP BY g.game_id, g.home_team_id
         ) AS subTable
     ORDER BY game_id, team_id;

CREATE UNIQUE INDEX team_game_uidx ON fix_stealing_a(team_id, game_id);

SELECT * FROM fix_stealing_a limit 0, 5;

DROP TABLE IF EXISTS team_batting_counts_fixed;
CREATE TABLE team_batting_counts_fixed LIKE team_batting_counts;
CREATE UNIQUE INDEX team_game_uidx ON team_batting_counts_fixed(team_id, game_id);
INSERT INTO team_batting_counts_fixed SELECT * FROM team_batting_counts;

INSERT INTO team_batting_counts_fixed (game_id, team_id, stolenBase2B, stolenBase3B, stolenBaseHome, caughtStealing2B, caughtStealing3B, caughtStealingHome)
SELECT game_id, team_id, stolenBase2B, stolenBase3B, stolenBaseHome, caughtStealing2B, caughtStealing3B, caughtStealingHome FROM fix_stealing_a fsa
ON DUPLICATE KEY UPDATE
    stolenBase2B         = fsa.stolenBase2B
    , stolenBase3B       = fsa.stolenBase3B
    , stolenBaseHome     = fsa.stolenBaseHome
    , caughtStealing2B   = fsa.caughtStealing2B
    , caughtStealing3B   = fsa.caughtStealing3B
    , caughtStealingHome = fsa.caughtStealingHome;

-- double check the results - they have values now
select stolenBase2B, count(*)   as Cnt from team_batting_counts_fixed group by stolenBase2B   order by Cnt desc;
select stolenBase3B, count(*)   as Cnt from team_batting_counts_fixed group by stolenBase3B   order by Cnt desc;
select stolenBaseHome, count(*) as Cnt from team_batting_counts_fixed group by stolenBaseHome order by Cnt desc;


-- Fixing similar features for team_pitching_counts table:
DROP TABLE IF EXISTS fix_stealing_p;
CREATE TABLE fix_stealing_p AS
SELECT * FROM
    (SELECT -- Away team
            g.game_id
            , g.away_team_id AS team_id
            , i.pitcher
            , SUM(CASE WHEN des = 'Stolen Base 2B'       THEN 1 ELSE 0 END) AS stolenBase2B
            , SUM(CASE WHEN des = 'Stolen Base 3B'       THEN 1 ELSE 0 END) AS stolenBase3B
            , SUM(CASE WHEN des = 'Stolen Base Home'     THEN 1 ELSE 0 END) AS stolenBaseHome
            , SUM(CASE WHEN des = 'Caught Stealing 2B'   THEN 1 ELSE 0 END) AS caughtStealing2B
            , SUM(CASE WHEN des = 'Caught Stealing 3B'   THEN 1 ELSE 0 END) AS caughtStealing3B
            , SUM(CASE WHEN des = 'Caught Stealing Home' THEN 1 ELSE 0 END) AS caughtStealingHome
         FROM inning i
         JOIN game g ON g.game_id = i.game_id
         WHERE i.half = 0 AND i.entry = 'runner'
         GROUP BY g.game_id, g.away_team_id, i.pitcher

     UNION
     SELECT -- Home team
            g.game_id
            , g.home_team_id AS team_id
            , i.pitcher
            , SUM(CASE WHEN des = 'Stolen Base 2B'       THEN 1 ELSE 0 END) AS stolenBase2B
            , SUM(CASE WHEN des = 'Stolen Base 3B'       THEN 1 ELSE 0 END) AS stolenBase3B
            , SUM(CASE WHEN des = 'Stolen Base Home'     THEN 1 ELSE 0 END) AS stolenBaseHome
            , SUM(CASE WHEN des = 'Caught Stealing 2B'   THEN 1 ELSE 0 END) AS caughtStealing2B
            , SUM(CASE WHEN des = 'Caught Stealing 3B'   THEN 1 ELSE 0 END) AS caughtStealing3B
            , SUM(CASE WHEN des = 'Caught Stealing Home' THEN 1 ELSE 0 END) AS caughtStealingHome
         FROM inning i
         JOIN game g ON g.game_id = i.game_id
         WHERE i.half = 1 AND i.entry = 'runner'
         GROUP BY g.game_id, g.home_team_id, i.pitcher
         ) AS subTable
     ORDER BY game_id, team_id;

CREATE UNIQUE INDEX team_game_pitcher_idx ON fix_stealing_p(team_id, game_id, pitcher);


DROP TABLE IF EXISTS team_pitching_counts_fixed;
CREATE TABLE team_pitching_counts_fixed LIKE team_pitching_counts;
CREATE UNIQUE INDEX team_game_uidx ON team_pitching_counts_fixed(team_id, game_id);
INSERT INTO team_pitching_counts_fixed SELECT * FROM team_pitching_counts;
select * from team_pitching_counts_fixed limit 0, 5;

INSERT INTO team_pitching_counts_fixed (game_id, team_id, stolenBase2B, stolenBase3B, stolenBaseHome, caughtStealing2B, caughtStealing3B, caughtStealingHome)
SELECT * FROM (SELECT game_id, team_id, stolenBase2B, stolenBase3B, stolenBaseHome, caughtStealing2B, caughtStealing3B, caughtStealingHome
                 FROM fix_stealing_p GROUP BY game_id, team_id
                 ) AS fsp
ON DUPLICATE KEY UPDATE
    stolenBase2B         = fsp.stolenBase2B
    , stolenBase3B       = fsp.stolenBase3B
    , stolenBaseHome     = fsp.stolenBaseHome
    , caughtStealing2B   = fsp.caughtStealing2B
    , caughtStealing3B   = fsp.caughtStealing3B
    , caughtStealingHome = fsp.caughtStealingHome;

-- double check the results - they have values now
select stolenBase2B,   count(*) as Cnt from team_pitching_counts_fixed group by stolenBase2B   order by Cnt desc;
select stolenBase3B,   count(*) as Cnt from team_pitching_counts_fixed group by stolenBase3B   order by Cnt desc;
select stolenBaseHome, count(*) as Cnt from team_pitching_counts_fixed group by stolenBaseHome order by Cnt desc;


/*****************************************************************************************************************
 *          CREATE ROLLING 100 DAYS, using the fixed team_batting_counts and fixed team_pitching_counts
/****************************************************************************************************************/

DROP TABLE IF EXISTS rolling_100_day;
CREATE TABLE rolling_100_day AS
SELECT
       tbc1.team_id
       , tbc1.game_id
       , COUNT(*)                                             AS Cnt
       , SUM(tbc2.plateApperance)                             AS plateApperance
       , SUM(tbc2.atBat)                                      AS atBat
       , SUM(tbc2.Hit)                                        AS Hit
       , SUM(tbc2.caughtStealing2B)                           AS caughtStealing2B
       , SUM(tbc2.caughtStealing3B)                           AS caughtStealing3B
       , SUM(tbc2.caughtStealingHome)                         AS caughtStealingHome
       , SUM(tbc2.stolenBase2B)                               AS stolenBase2B
       , SUM(tbc2.stolenBase3B)                               AS stolenBase3B
       , SUM(tbc2.stolenBaseHome)                             AS stolenBaseHome
       , SUM(tbc2.toBase)                                     AS toBase
       , SUM(tbc2.Batter_Interference)                        AS Batter_Interference
       , SUM(tbc2.Bunt_Ground_Out) + SUM(tbc2.Bunt_Groundout) AS Bunt_Ground_Out
       , SUM(tbc2.Bunt_Pop_Out)                               AS Bunt_Pop_Out
       , SUM(tbc2.Catcher_Interference)                       AS Catcher_Interference
       , SUM(tbc2.`Double`)                                   AS `Double`
       , SUM(tbc2.Double_Play)                                AS Double_Play
       , SUM(tbc2.Fan_interference)                           AS Fan_interference
       , SUM(tbc2.Field_Error)                                AS Field_Error
       , SUM(tbc2.Fielders_Choice)                            AS Fielders_Choice
       , SUM(tbc2.Fielders_Choice_Out)                        AS Fielders_Choice_Out
       , SUM(tbc2.Fly_Out) + SUM(tbc2.Flyout)                 AS Fly_Out
       , SUM(tbc2.Force_Out) + SUM(tbc2.Forceout)             AS Force_Out
       , SUM(tbc2.Ground_Out) + SUM(tbc2.Groundout)           AS Ground_Out
       , SUM(tbc2.Grounded_Into_DP)                           AS Grounded_Into_DP
       , SUM(tbc2.Hit_By_Pitch)                               AS Hit_By_Pitch
       , SUM(tbc2.Home_Run)                                   AS Home_Run
       , SUM(tbc2.Intent_Walk)                                AS Intent_Walk
       , SUM(tbc2.Line_Out) + SUM(tbc2.Lineout)               AS Line_Out
       , SUM(tbc2.Pop_Out)                                    AS Pop_Out
       , SUM(tbc2.Runner_Out)                                 AS Runner_Out
       , SUM(tbc2.Sac_Bunt)                                   AS Sac_Bunt
       , SUM(tbc2.Sac_Fly)                                    AS Sac_Fly
       , SUM(tbc2.Sac_Fly_DP)                                 AS Sac_Fly_DP
       , SUM(tbc2.Sacrifice_Bunt_DP)                          AS Sacrifice_Bunt_DP
       , SUM(tbc2.Single)                                     AS Single
       , SUM(tbc2.`Strikeout_-_DP`)                           AS Strikeout_DP
       , SUM(tbc2.`Strikeout_-_TP`)                           AS Strikeout_TP
       , SUM(tbc2.Triple)                                     AS Triple
       , SUM(tbc2.Triple_Play)                                AS Triple_Play
       , SUM(tbc2.Walk)                                       AS Walk
--       , SUM(pc.DaysSinceLastPitch)                           AS DaysSinceLastPitch
--        , SUM(pc.pitchesThrown)                                AS pitchesThrown
--        , SUM(pc.endingInning - pc.startingInning)             AS inningPlayed
--        , SUM(pc.bullpenPitcher)                               AS bullpenPitcher
--        , SUM(pc.startingPitcher)                              AS startingPitcher
--        , SUM(tpc.bullpenOutsPlayed)                           AS bullpenOutsPlayed
--        , SUM(tpc.bullpenWalk)                                 AS bullpenWalk
--        , SUM(tpc.bullpenHit)                                  AS bullpenHit
--        , COUNT(pig.pitcher)                                   AS pitcher_count
      , SUM(tpc.plateApperance)                             AS plateApperance2
       , SUM(tpc.atBat)                                      AS atBat2
       , SUM(tpc.Hit)                                        AS Hit2
       , SUM(tpc.caughtStealing2B)                           AS caughtStealing2B2
       , SUM(tpc.caughtStealing3B)                           AS caughtStealing3B2
       , SUM(tpc.caughtStealingHome)                         AS caughtStealingHome2
       , SUM(tpc.stolenBase2B)                               AS stolenBase2B2
       , SUM(tpc.stolenBase3B)                               AS stolenBase3B2
       , SUM(tpc.stolenBaseHome)                             AS stolenBaseHome2
       , SUM(tpc.toBase)                                     AS toBase2
       , SUM(tpc.Batter_Interference)                        AS Batter_Interference2
       , SUM(tpc.Bunt_Ground_Out) + SUM(tbc2.Bunt_Groundout) AS Bunt_Ground_Out2
       , SUM(tpc.Bunt_Pop_Out)                               AS Bunt_Pop_Out2
       , SUM(tpc.Catcher_Interference)                       AS Catcher_Interference2
       , SUM(tpc.`Double`)                                   AS `Double2`
       , SUM(tpc.Double_Play)                                AS Double_Play2
       , SUM(tpc.Fan_interference)                           AS Fan_interference2
       , SUM(tpc.Field_Error)                                AS Field_Error2
       , SUM(tpc.Fielders_Choice)                            AS Fielders_Choice2
       , SUM(tpc.Fielders_Choice_Out)                        AS Fielders_Choice_Out2
       , SUM(tpc.Fly_Out) + SUM(tbc2.Flyout)                 AS Fly_Out2
       , SUM(tpc.Force_Out) + SUM(tbc2.Forceout)             AS Force_Out2
       , SUM(tpc.Ground_Out) + SUM(tbc2.Groundout)           AS Ground_Out2
       , SUM(tpc.Grounded_Into_DP)                           AS Grounded_Into_DP2
       , SUM(tpc.Hit_By_Pitch)                               AS Hit_By_Pitch2
       , SUM(tpc.Home_Run)                                   AS Home_Run2
       , SUM(tpc.Intent_Walk)                                AS Intent_Walk2
       , SUM(tpc.Line_Out) + SUM(tbc2.Lineout)               AS Line_Out2
       , SUM(tpc.Pop_Out)                                    AS Pop_Out2
       , SUM(tpc.Runner_Out)                                 AS Runner_Out2
       , SUM(tpc.Sac_Bunt)                                   AS Sac_Bunt2
       , SUM(tpc.Sac_Fly)                                    AS Sac_Fly2
       , SUM(tpc.Sac_Fly_DP)                                 AS Sac_Fly_DP2
       , SUM(tpc.Sacrifice_Bunt_DP)                          AS Sacrifice_Bunt_DP2
       , SUM(tpc.Single)                                     AS Single2
       , SUM(tpc.`Strikeout_-_DP`)                           AS Strikeout_DP2
       , SUM(tpc.`Strikeout_-_TP`)                           AS Strikeout_TP2
       , SUM(tpc.Triple)                                     AS Triple2
       , SUM(tpc.Triple_Play)                                AS Triple_Play2
       , SUM(tpc.Walk)                                       AS Walk2
    FROM team_batting_counts_fixed tbc1
    JOIN team_batting_counts_fixed tbc2 ON tbc1.team_id = tbc2.team_id
    JOIN game g1                        ON tbc1.game_id = g1.game_id  AND g1.type IN ("R")
    JOIN game g2                        ON tbc2.game_id = g2.game_id  AND g2.type IN ("R") AND
              g2.local_date BETWEEN DATE_SUB(g1.local_date, INTERVAL 100 DAY) AND g1.local_date
--     JOIN pitcher_counts        pc  ON tbc1.team_id = pc.team_id  AND tbc1.game_id = pc.game_id
--     JOIN pitchersInGame        pig ON tbc1.team_id = pig.team_id AND tbc1.game_id = pig.game_id
    JOIN team_pitching_counts_fixed  tpc ON tbc1.team_id = tpc.team_id AND tbc1.game_id = tpc.game_id
    GROUP BY tbc1.team_id, tbc1.game_id
    ORDER BY tbc1.team_id;

CREATE UNIQUE INDEX team_game ON rolling_100_day(team_id, game_id);


/*****************************************************************************************************************
 *          CREATE ROLLING 200 DAYS, using the fixed team_batting_counts and fixed team_pitching_counts
/****************************************************************************************************************/

DROP TABLE IF EXISTS rolling_200_day;
CREATE TABLE rolling_200_day AS
SELECT
       tbc1.team_id
       , tbc1.game_id
       , COUNT(*)                                             AS Cnt
       , SUM(tbc2.plateApperance)                             AS plateApperance
       , SUM(tbc2.atBat)                                      AS atBat
       , SUM(tbc2.Hit)                                        AS Hit
       , SUM(tbc2.caughtStealing2B)                           AS caughtStealing2B
       , SUM(tbc2.caughtStealing3B)                           AS caughtStealing3B
       , SUM(tbc2.caughtStealingHome)                         AS caughtStealingHome
       , SUM(tbc2.stolenBase2B)                               AS stolenBase2B
       , SUM(tbc2.stolenBase3B)                               AS stolenBase3B
       , SUM(tbc2.stolenBaseHome)                             AS stolenBaseHome
       , SUM(tbc2.toBase)                                     AS toBase
       , SUM(tbc2.Batter_Interference)                        AS Batter_Interference
       , SUM(tbc2.Bunt_Ground_Out) + SUM(tbc2.Bunt_Groundout) AS Bunt_Ground_Out
       , SUM(tbc2.Bunt_Pop_Out)                               AS Bunt_Pop_Out
       , SUM(tbc2.Catcher_Interference)                       AS Catcher_Interference
       , SUM(tbc2.`Double`)                                   AS `Double`
       , SUM(tbc2.Double_Play)                                AS Double_Play
       , SUM(tbc2.Fan_interference)                           AS Fan_interference
       , SUM(tbc2.Field_Error)                                AS Field_Error
       , SUM(tbc2.Fielders_Choice)                            AS Fielders_Choice
       , SUM(tbc2.Fielders_Choice_Out)                        AS Fielders_Choice_Out
       , SUM(tbc2.Fly_Out) + SUM(tbc2.Flyout)                 AS Fly_Out
       , SUM(tbc2.Force_Out) + SUM(tbc2.Forceout)             AS Force_Out
       , SUM(tbc2.Ground_Out) + SUM(tbc2.Groundout)           AS Ground_Out
       , SUM(tbc2.Grounded_Into_DP)                           AS Grounded_Into_DP
       , SUM(tbc2.Hit_By_Pitch)                               AS Hit_By_Pitch
       , SUM(tbc2.Home_Run)                                   AS Home_Run
       , SUM(tbc2.Intent_Walk)                                AS Intent_Walk
       , SUM(tbc2.Line_Out) + SUM(tbc2.Lineout)               AS Line_Out
       , SUM(tbc2.Pop_Out)                                    AS Pop_Out
       , SUM(tbc2.Runner_Out)                                 AS Runner_Out
       , SUM(tbc2.Sac_Bunt)                                   AS Sac_Bunt
       , SUM(tbc2.Sac_Fly)                                    AS Sac_Fly
       , SUM(tbc2.Sac_Fly_DP)                                 AS Sac_Fly_DP
       , SUM(tbc2.Sacrifice_Bunt_DP)                          AS Sacrifice_Bunt_DP
       , SUM(tbc2.Single)                                     AS Single
       , SUM(tbc2.`Strikeout_-_DP`)                           AS Strikeout_DP
       , SUM(tbc2.`Strikeout_-_TP`)                           AS Strikeout_TP
       , SUM(tbc2.Triple)                                     AS Triple
       , SUM(tbc2.Triple_Play)                                AS Triple_Play
       , SUM(tbc2.Walk)                                       AS Walk
--        , SUM(pc.DaysSinceLastPitch)                           AS DaysSinceLastPitch
--        , SUM(pc.pitchesThrown)                                AS pitchesThrown
--        , SUM(pc.endingInning - pc.startingInning)             AS inningPlayed
--        , SUM(pc.bullpenPitcher)                               AS bullpenPitcher
--        , SUM(pc.startingPitcher)                              AS startingPitcher
--        , SUM(tpc.bullpenOutsPlayed)                           AS bullpenOutsPlayed
--        , SUM(tpc.bullpenWalk)                                 AS bullpenWalk
--        , SUM(tpc.bullpenHit)                                  AS bullpenHit
--        , COUNT(pig.pitcher)                                   AS pitcher_count
      , SUM(tpc.plateApperance)                             AS plateApperance2
       , SUM(tpc.atBat)                                      AS atBat2
       , SUM(tpc.Hit)                                        AS Hit2
       , SUM(tpc.caughtStealing2B)                           AS caughtStealing2B2
       , SUM(tpc.caughtStealing3B)                           AS caughtStealing3B2
       , SUM(tpc.caughtStealingHome)                         AS caughtStealingHome2
       , SUM(tpc.stolenBase2B)                               AS stolenBase2B2
       , SUM(tpc.stolenBase3B)                               AS stolenBase3B2
       , SUM(tpc.stolenBaseHome)                             AS stolenBaseHome2
       , SUM(tpc.toBase)                                     AS toBase2
       , SUM(tpc.Batter_Interference)                        AS Batter_Interference2
       , SUM(tpc.Bunt_Ground_Out) + SUM(tbc2.Bunt_Groundout) AS Bunt_Ground_Out2
       , SUM(tpc.Bunt_Pop_Out)                               AS Bunt_Pop_Out2
       , SUM(tpc.Catcher_Interference)                       AS Catcher_Interference2
       , SUM(tpc.`Double`)                                   AS `Double2`
       , SUM(tpc.Double_Play)                                AS Double_Play2
       , SUM(tpc.Fan_interference)                           AS Fan_interference2
       , SUM(tpc.Field_Error)                                AS Field_Error2
       , SUM(tpc.Fielders_Choice)                            AS Fielders_Choice2
       , SUM(tpc.Fielders_Choice_Out)                        AS Fielders_Choice_Out2
       , SUM(tpc.Fly_Out) + SUM(tbc2.Flyout)                 AS Fly_Out2
       , SUM(tpc.Force_Out) + SUM(tbc2.Forceout)             AS Force_Out2
       , SUM(tpc.Ground_Out) + SUM(tbc2.Groundout)           AS Ground_Out2
       , SUM(tpc.Grounded_Into_DP)                           AS Grounded_Into_DP2
       , SUM(tpc.Hit_By_Pitch)                               AS Hit_By_Pitch2
       , SUM(tpc.Home_Run)                                   AS Home_Run2
       , SUM(tpc.Intent_Walk)                                AS Intent_Walk2
       , SUM(tpc.Line_Out) + SUM(tbc2.Lineout)               AS Line_Out2
       , SUM(tpc.Pop_Out)                                    AS Pop_Out2
       , SUM(tpc.Runner_Out)                                 AS Runner_Out2
       , SUM(tpc.Sac_Bunt)                                   AS Sac_Bunt2
       , SUM(tpc.Sac_Fly)                                    AS Sac_Fly2
       , SUM(tpc.Sac_Fly_DP)                                 AS Sac_Fly_DP2
       , SUM(tpc.Sacrifice_Bunt_DP)                          AS Sacrifice_Bunt_DP2
       , SUM(tpc.Single)                                     AS Single2
       , SUM(tpc.`Strikeout_-_DP`)                           AS Strikeout_DP2
       , SUM(tpc.`Strikeout_-_TP`)                           AS Strikeout_TP2
       , SUM(tpc.Triple)                                     AS Triple2
       , SUM(tpc.Triple_Play)                                AS Triple_Play2
       , SUM(tpc.Walk)                                       AS Walk2
    FROM team_batting_counts_fixed tbc1
    JOIN team_batting_counts_fixed tbc2 ON tbc1.team_id = tbc2.team_id
    JOIN game g1                        ON tbc1.game_id = g1.game_id  AND g1.type IN ("R")
    JOIN game g2                        ON tbc2.game_id = g2.game_id  AND g2.type IN ("R") AND
             g2.local_date BETWEEN DATE_SUB(g1.local_date, INTERVAL 200 DAY) AND g1.local_date
--     JOIN pitcher_counts pc ON tbc1.team_id = pc.team_id AND tbc1.game_id = pc.game_id
--     JOIN pitchersInGame pig ON tbc1.team_id = pig.team_id AND tbc1.game_id = pig.game_id
    JOIN team_pitching_counts_fixed tpc ON tbc1.team_id = tpc.team_id AND tbc1.game_id = tpc.game_id
    GROUP BY tbc1.team_id, tbc1.game_id
    ORDER BY tbc1.team_id;

CREATE UNIQUE INDEX team_game ON rolling_200_day(team_id, game_id);


/*****************************************************************************************************************
 *          CREATE ROLLING 365 DAYS, using the fixed team_batting_counts and fixed team_pitching_counts
/****************************************************************************************************************/

DROP TABLE IF EXISTS rolling_365_day; -- 4 minutes running
CREATE TABLE rolling_365_day AS
SELECT
       tbc1.team_id
       , tbc1.game_id
       , COUNT(*)                                             AS Cnt
       , SUM(tbc2.plateApperance)                             AS plateApperance
       , SUM(tbc2.atBat)                                      AS atBat
       , SUM(tbc2.Hit)                                        AS Hit
       , SUM(tbc2.caughtStealing2B)                           AS caughtStealing2B
       , SUM(tbc2.caughtStealing3B)                           AS caughtStealing3B
       , SUM(tbc2.caughtStealingHome)                         AS caughtStealingHome
       , SUM(tbc2.stolenBase2B)                               AS stolenBase2B
       , SUM(tbc2.stolenBase3B)                               AS stolenBase3B
       , SUM(tbc2.stolenBaseHome)                             AS stolenBaseHome
       , SUM(tbc2.toBase)                                     AS toBase
       , SUM(tbc2.Batter_Interference)                        AS Batter_Interference
       , SUM(tbc2.Bunt_Ground_Out) + SUM(tbc2.Bunt_Groundout) AS Bunt_Ground_Out
       , SUM(tbc2.Bunt_Pop_Out)                               AS Bunt_Pop_Out
       , SUM(tbc2.Catcher_Interference)                       AS Catcher_Interference
       , SUM(tbc2.`Double`)                                   AS `Double`
       , SUM(tbc2.Double_Play)                                AS Double_Play
       , SUM(tbc2.Fan_interference)                           AS Fan_interference
       , SUM(tbc2.Field_Error)                                AS Field_Error
       , SUM(tbc2.Fielders_Choice)                            AS Fielders_Choice
       , SUM(tbc2.Fielders_Choice_Out)                        AS Fielders_Choice_Out
       , SUM(tbc2.Fly_Out) + SUM(tbc2.Flyout)                 AS Fly_Out
       , SUM(tbc2.Force_Out) + SUM(tbc2.Forceout)             AS Force_Out
       , SUM(tbc2.Ground_Out) + SUM(tbc2.Groundout)           AS Ground_Out
       , SUM(tbc2.Grounded_Into_DP)                           AS Grounded_Into_DP
       , SUM(tbc2.Hit_By_Pitch)                               AS Hit_By_Pitch
       , SUM(tbc2.Home_Run)                                   AS Home_Run
       , SUM(tbc2.Intent_Walk)                                AS Intent_Walk
       , SUM(tbc2.Line_Out) + SUM(tbc2.Lineout)               AS Line_Out
       , SUM(tbc2.Pop_Out)                                    AS Pop_Out
       , SUM(tbc2.Runner_Out)                                 AS Runner_Out
       , SUM(tbc2.Sac_Bunt)                                   AS Sac_Bunt
       , SUM(tbc2.Sac_Fly)                                    AS Sac_Fly
       , SUM(tbc2.Sac_Fly_DP)                                 AS Sac_Fly_DP
       , SUM(tbc2.Sacrifice_Bunt_DP)                          AS Sacrifice_Bunt_DP
       , SUM(tbc2.Single)                                     AS Single
       , SUM(tbc2.`Strikeout_-_DP`)                           AS Strikeout_DP
       , SUM(tbc2.`Strikeout_-_TP`)                           AS Strikeout_TP
       , SUM(tbc2.Triple)                                     AS Triple
       , SUM(tbc2.Triple_Play)                                AS Triple_Play
       , SUM(tbc2.Walk)                                       AS Walk
--        , SUM(pc.DaysSinceLastPitch)                           AS DaysSinceLastPitch
--        , SUM(pc.pitchesThrown)                                AS pitchesThrown
--        , SUM(pc.endingInning - pc.startingInning)             AS inningPlayed
--        , SUM(pc.bullpenPitcher)                               AS bullpenPitcher
--        , SUM(pc.startingPitcher)                              AS startingPitcher
--        , SUM(tpc.bullpenOutsPlayed)                           AS bullpenOutsPlayed
--        , SUM(tpc.bullpenWalk)                                 AS bullpenWalk
--        , SUM(tpc.bullpenHit)                                  AS bullpenHit
--        , COUNT(pig.pitcher)                                   AS pitcher_count
      , SUM(tpc.plateApperance)                             AS plateApperance2
       , SUM(tpc.atBat)                                      AS atBat2
       , SUM(tpc.Hit)                                        AS Hit2
       , SUM(tpc.caughtStealing2B)                           AS caughtStealing2B2
       , SUM(tpc.caughtStealing3B)                           AS caughtStealing3B2
       , SUM(tpc.caughtStealingHome)                         AS caughtStealingHome2
       , SUM(tpc.stolenBase2B)                               AS stolenBase2B2
       , SUM(tpc.stolenBase3B)                               AS stolenBase3B2
       , SUM(tpc.stolenBaseHome)                             AS stolenBaseHome2
       , SUM(tpc.toBase)                                     AS toBase2
       , SUM(tpc.Batter_Interference)                        AS Batter_Interference2
       , SUM(tpc.Bunt_Ground_Out) + SUM(tbc2.Bunt_Groundout) AS Bunt_Ground_Out2
       , SUM(tpc.Bunt_Pop_Out)                               AS Bunt_Pop_Out2
       , SUM(tpc.Catcher_Interference)                       AS Catcher_Interference2
       , SUM(tpc.`Double`)                                   AS `Double2`
       , SUM(tpc.Double_Play)                                AS Double_Play2
       , SUM(tpc.Fan_interference)                           AS Fan_interference2
       , SUM(tpc.Field_Error)                                AS Field_Error2
       , SUM(tpc.Fielders_Choice)                            AS Fielders_Choice2
       , SUM(tpc.Fielders_Choice_Out)                        AS Fielders_Choice_Out2
       , SUM(tpc.Fly_Out) + SUM(tbc2.Flyout)                 AS Fly_Out2
       , SUM(tpc.Force_Out) + SUM(tbc2.Forceout)             AS Force_Out2
       , SUM(tpc.Ground_Out) + SUM(tbc2.Groundout)           AS Ground_Out2
       , SUM(tpc.Grounded_Into_DP)                           AS Grounded_Into_DP2
       , SUM(tpc.Hit_By_Pitch)                               AS Hit_By_Pitch2
       , SUM(tpc.Home_Run)                                   AS Home_Run2
       , SUM(tpc.Intent_Walk)                                AS Intent_Walk2
       , SUM(tpc.Line_Out) + SUM(tbc2.Lineout)               AS Line_Out2
       , SUM(tpc.Pop_Out)                                    AS Pop_Out2
       , SUM(tpc.Runner_Out)                                 AS Runner_Out2
       , SUM(tpc.Sac_Bunt)                                   AS Sac_Bunt2
       , SUM(tpc.Sac_Fly)                                    AS Sac_Fly2
       , SUM(tpc.Sac_Fly_DP)                                 AS Sac_Fly_DP2
       , SUM(tpc.Sacrifice_Bunt_DP)                          AS Sacrifice_Bunt_DP2
       , SUM(tpc.Single)                                     AS Single2
       , SUM(tpc.`Strikeout_-_DP`)                           AS Strikeout_DP2
       , SUM(tpc.`Strikeout_-_TP`)                           AS Strikeout_TP2
       , SUM(tpc.Triple)                                     AS Triple2
       , SUM(tpc.Triple_Play)                                AS Triple_Play2
       , SUM(tpc.Walk)                                       AS Walk2
    FROM team_batting_counts_fixed tbc1
    JOIN team_batting_counts_fixed tbc2 ON tbc1.team_id = tbc2.team_id
    JOIN game g1                        ON tbc1.game_id = g1.game_id  AND g1.type IN ("R")
    JOIN game g2                        ON tbc2.game_id = g2.game_id  AND g2.type IN ("R") AND
             g2.local_date BETWEEN DATE_SUB(g1.local_date, INTERVAL 365 DAY) AND g1.local_date
--     JOIN pitcher_counts pc ON tbc1.team_id = pc.team_id AND tbc1.game_id = pc.game_id
--     JOIN pitchersInGame pig ON tbc1.team_id = pig.team_id AND tbc1.game_id = pig.game_id
    JOIN team_pitching_counts_fixed tpc ON tbc1.team_id = tpc.team_id AND tbc1.game_id = tpc.game_id
    GROUP BY tbc1.team_id, tbc1.game_id
    ORDER BY tbc1.team_id;

CREATE UNIQUE INDEX team_game ON rolling_365_day(team_id, game_id);

SELECT
       game_id
       , home_throwinghand
       , away_throwinghand
    FROM pregame
    order by game_id;

select Field_Error , count(*) as Cnt from rolling_365_day group by Field_Error order by Cnt desc;
select plateApperance , count(*) as Cnt from rolling_365_day group by plateApperance order by Cnt desc;


/*****************************************************************************************************************
 *                                              CREATE NEW FEATURES
/****************************************************************************************************************/

DROP TABLE IF EXISTS finalProject_data2;
CREATE TABLE finalProject_data2 AS
SELECT
       g.game_id
       , CASE
            WHEN b.winner_home_or_away = 'H' THEN 1
            WHEN b.winner_home_or_away = 'A' THEN 0
            ELSE 0  END         AS home_team_wins
       , g.home_team_id         AS home_team_id
       , g.away_team_id         AS away_team_id
       , b.temp                               AS temp
       , b.overcast                           AS overcast
       , b.wind                               AS wind
       , b.winddir                            AS winddir
       , b.home_hits / NULLIF(b.away_hits, 0) - 1.0      AS hits_ratio
       , b.home_hits - b.away_hits                       AS hits_diff
       , b.home_streak / NULLIF(b.away_streak, 0) - 1.0  AS streak_ratio
       , b.home_streak - b.away_streak                   AS streak_diff
       , s.name                                          AS stadium_name                                      -- 9
       , p.venue_short                                   AS venue_short
       , p.status                                        AS status
       , p.ampm                                          AS ampm
       , p.league                                        AS league
       , p.series_game_number                            AS series_game_number
--        , p.home_number / CAST(p.away_number AS FLOAT) - 1.0  AS number_ratio  -- truncated error
--        , p.home_number - p.away_number        AS number_diff                  -- truncated error
       , p.home_throwinghand                  AS home_throwinghand
       , p.away_throwinghand                  AS away_throwinghand                                            -- 18
       , r1dh.Hit / r1dh.atBat                                              AS BA_1home
       , r1da.Hit / NULLIF(r1da.atBat, 0)                                   AS BA_1away
       , r2dh.Hit / NULLIF(r2dh.atBat, 0)                                   AS BA_2home
       , r2da.Hit / NULLIF(r2da.atBat, 0)                                   AS BA_2away
       , r3dh.Hit / NULLIF(r3dh.atBat, 0)                                   AS BA_3home
       , r3da.Hit / NULLIF(r3da.atBat, 0)                                   AS BA_3away
       , (r1dh.Hit / NULLIF(r1dh.atBat, 0)) / (r1da.Hit / NULLIF(r1da.atBat, 0)) - 1.0 AS BA_1ratio           -- 18
       , (r1dh.Hit / NULLIF(r1dh.atBat, 0)) - (r1da.Hit / NULLIF(r1da.atBat, 0))       AS BA_1diff
       , (r2dh.Hit / NULLIF(r2dh.atBat, 0)) / (r2da.Hit / NULLIF(r2da.atBat, 0)) - 1.0 AS BA_2ratio
       , (r2dh.Hit / NULLIF(r2dh.atBat, 0)) - (r2da.Hit / NULLIF(r2da.atBat, 0))       AS BA_2diff
       , (r3dh.Hit / NULLIF(r3dh.atBat, 0)) / (r3da.Hit / NULLIF(r3da.atBat, 0)) - 1.0 AS BA_3ratio
       , (r3dh.Hit / NULLIF(r3dh.atBat, 0)) - (r3da.Hit / NULLIF(r3da.atBat, 0))       AS BA_3diff            -- 24
       , ((r1dh.Single + 2*r1dh.`Double` + 3*r1dh.Triple + 4*r1dh.Home_Run)/r1dh.atBat) /
         ((r1da.Single + 2*r1da.`Double` + 3*r1da.Triple + 4*r1da.Home_Run)/r1da.atBat) - 1.0 AS Slug_1ratio
       , ((r1dh.Single + 2*r1dh.`Double` + 3*r1dh.Triple + 4*r1dh.Home_Run)/r1dh.atBat) -
         ((r1da.Single + 2*r1da.`Double` + 3*r1da.Triple + 4*r1da.Home_Run)/r1da.atBat)       AS Slug_1diff
       , ((r2dh.Single + 2*r2dh.`Double` + 3*r2dh.Triple + 4*r2dh.Home_Run)/r2dh.atBat) /
         ((r2da.Single + 2*r2da.`Double` + 3*r2da.Triple + 4*r2da.Home_Run)/r2da.atBat) - 1.0 AS Slug_2ratio
       , ((r2dh.Single + 2*r2dh.`Double` + 3*r2dh.Triple + 4*r2dh.Home_Run)/r2dh.atBat) -
         ((r2da.Single + 2*r2da.`Double` + 3*r2da.Triple + 4*r2da.Home_Run)/r2da.atBat)       AS Slug_2diff
       , ((r3dh.Single + 2*r3dh.`Double` + 3*r3dh.Triple + 4*r3dh.Home_Run)/r3dh.atBat) /
         ((r3da.Single + 2*r3da.`Double` + 3*r3da.Triple + 4*r3da.Home_Run)/r3da.atBat) - 1.0 AS Slug_3ratio
       , ((r3dh.Single + 2*r3dh.`Double` + 3*r3dh.Triple + 4*r3dh.Home_Run)/r3dh.atBat) -
         ((r3da.Single + 2*r3da.`Double` + 3*r3da.Triple + 4*r3da.Home_Run)/r3da.atBat)       AS Slug_3diff   -- 30
       , r1dh.Field_Error  / r1da.Field_Error  - 1.0        AS Field_Error_ratio
       , r1dh.Field_Error2  / r1da.Field_Error2  - 1.0      AS Field_Error_ratio2
       , r1dh.Field_Error  - r1da.Field_Error               AS Field_Error_diff
       , r1dh.Field_Error2 - r1da.Field_Error2              AS Field_Error_diff2                              -- 34
       , r1dh.plateApperance  / r1da.plateApperance  - 1.0  AS plateApperance_ratio
       , r1dh.plateApperance2 / r1da.plateApperance2 - 1.0  AS plateApperance_ratio2
       , r1dh.plateApperance  - r1da.plateApperance         AS plateApperance_diff
       , r1dh.plateApperance2 - r1da.plateApperance2        AS plateApperance_diff2                           -- 38
    FROM game g
    JOIN rolling_100_day r1dh ON g.game_id = r1dh.game_id AND g.home_team_id = r1dh.team_id
    JOIN rolling_100_day r1da ON g.game_id = r1da.game_id AND g.away_team_id = r1da.team_id
    JOIN rolling_200_day r2dh ON g.game_id = r2dh.game_id AND g.home_team_id = r2dh.team_id
    JOIN rolling_200_day r2da ON g.game_id = r2da.game_id AND g.away_team_id = r2da.team_id
    JOIN rolling_365_day r3dh ON g.game_id = r3dh.game_id AND g.home_team_id = r3dh.team_id
    JOIN rolling_365_day r3da ON g.game_id = r3da.game_id AND g.away_team_id = r3da.team_id
    JOIN boxscore b ON g.game_id = b.game_id                      -- 16,000 ROWS
    JOIN stadium s  ON g.stadium_id = s.stadium_id                -- 86 ROWS
    JOIN pregame p  ON g.game_id = p.game_id                      -- 14,000 ROWS
    WHERE CAST(r1da.Field_Error AS INT) > 0    AND
          CAST(r1da.Field_Error2 AS INT) > 0   AND
          r1da.plateApperance  > 0 AND r1da.plateApperance2 > 0
    ORDER BY g.game_id;

-- a feature -> 12 features:  * 2 (for batters and pitchers) * 3 (for 100, 200, 365) * 2 (for home team, away team)

CREATE INDEX game_idx ON finalProject_data2(game_id);


/*********************************
 *  UPDATE table for Python Code
 *********************************/
ALTER TABLE finalProject_data2 CHANGE home_team_wins HomeTeamWins int;


DROP TABLE IF EXISTS finalProject_data_addedDate;
CREATE TABLE finalProject_data_addedDate AS
SELECT
       fpd.*
       , g.local_date
    FROM finalProject_data2 fpd
    JOIN game              g     ON fpd.game_id = g.game_id;











