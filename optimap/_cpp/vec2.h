#pragma once
class Vec2
	{	
	public:
		float f[2];
		
		Vec2(float x, float y)
		{
			f[0] =x;
			f[1] =y;
		}
		
		Vec2() {}
		
		float length() const
		{
			return sqrt(f[0]*f[0]+f[1]*f[1]);
		}
		
		Vec2 normalized() const
		{
			float l = length();
			return Vec2(f[0]/l,f[1]/l);
		}
		
		void operator+= (const Vec2 &v)
		{
			f[0]+=v.f[0];
			f[1]+=v.f[1];
		}
		
		Vec2 operator/ (const float &a)
		{
			return Vec2(f[0]/a,f[1]/a);
		}
		
		Vec2 operator- (const Vec2 &v)
		{
			return Vec2(f[0]-v.f[0],f[1]-v.f[1]);
		}
		
		Vec2 operator+ (const Vec2 &v)
		{
			return Vec2(f[0]+v.f[0],f[1]+v.f[1]);
		}
		
		Vec2 operator* (const float &a)
		{
			return Vec2(f[0]*a,f[1]*a);
		}
		
		Vec2 operator-()
		{
			return Vec2(-f[0],-f[1]);
		}
		
		//Vec2 cross(const Vec2 &v)
		//{
		//	return Vec3(f[1]*v.f[2] - f[2]*v.f[1], f[2]*v.f[0] - f[0]*v.f[2], f[0]*v.f[1] - f[1]*v.f[0]);
		//}
		
		float dot(const Vec2 &v)
		{
			return f[0]*v.f[0] + f[1]*v.f[1];
		}
	};