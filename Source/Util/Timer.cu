#include "Timer.cuh"

#include <SDL2/SDL.h>

Timer::Timer()
{
    m_startTicks = 0;
    m_pausedTicks = 0;
    m_paused = false;
    m_started = false;
}

void Timer::start()
{
    m_started = true;
    m_paused = false;

    m_startTicks = SDL_GetTicks();
}

void Timer::stop()
{
    m_started = false;
    m_paused = false;
}

void Timer::pause()
{
    if(m_started && !m_paused)
    {
        m_paused = true;
        m_pausedTicks = SDL_GetTicks() - m_startTicks;
    }
}

void Timer::unpause()
{
    if( m_paused == true )
    {
        m_paused = false;

        m_startTicks = SDL_GetTicks() - m_pausedTicks;
        m_pausedTicks = 0;
    }
}

int Timer::get_ticks()
{
    if( m_started == true )
    {
        if( m_paused == true )
        {
            return m_pausedTicks;
        }
        else
        {
            return SDL_GetTicks() - m_startTicks;
        }
    }

    return 0;
}