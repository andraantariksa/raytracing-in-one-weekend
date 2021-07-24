#ifndef RAYTRACING_SOURCE_UTIL_TIMER_CUH
#define RAYTRACING_SOURCE_UTIL_TIMER_CUH

class Timer
{
private:
    int m_startTicks;
    int m_pausedTicks;
    bool m_paused;
    bool m_started;
public:
    Timer();

    void start();
    void stop();
    void pause();
    void unpause();
    int get_ticks();
    [[nodiscard]]
    bool is_started() const { return m_started; };
    [[nodiscard]]
    bool is_paused() const { return m_paused; };
};

#endif //RAYTRACING_SOURCE_UTIL_TIMER_CUH
